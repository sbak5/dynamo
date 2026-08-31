// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

mod default;
mod policy;

pub use default::DefaultWorkerSelector;

use default::{DefaultWorkerPicker, DefaultWorkerScorer};
pub use policy::{
    ScoredWorkerCandidate, WorkerCacheInput, WorkerCandidate, WorkerFilter, WorkerInputView,
    WorkerInputs, WorkerLoadInput, WorkerPicker, WorkerScorer, WorkerSelectionContext,
    WorkerSelectionPolicy,
};

use default::{pick_default_worker, selection_weights};
use policy::{
    CustomWorkerSelectionState, WorkerSelectionPolicyStateRef, collect_custom_candidates,
};

use super::config::KvRouterConfig;
use super::filter::{RoutingEligibility, WorkerEligibilityError};
use super::types::{KvSchedulerError, SchedulingRequest, WorkerSelectionPolicyError};
use crate::protocols::{WorkerConfigLike, WorkerId, WorkerSelectionResult, WorkerWithDpRank};

/// Low-level selector used by routing hosts.
///
/// External policies should use [`WorkerSelectionPolicy`].
pub trait WorkerSelector<C: WorkerConfigLike> {
    /// Optional worker data required by this selector.
    fn required_worker_inputs(&self) -> WorkerInputs;

    /// Whether an eligible affinity target exclusively constrains worker selection.
    ///
    /// The default selector uses exclusive affinity. Custom policies receive affinity as
    /// advisory context and may choose another eligible worker.
    fn uses_exclusive_affinity_target(&self) -> bool {
        false
    }

    fn select_worker(
        &self,
        input: WorkerSelectionInput<'_, C>,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;

    /// Host-only lifecycle hook. Custom selectors keep the ordinary
    /// `select_worker` contract unless they opt into a decision summary.
    #[doc(hidden)]
    fn select_worker_with_lifecycle(
        &self,
        input: WorkerSelectionInput<'_, C>,
        _span: Option<&tracing::Span>,
        _investigation: bool,
    ) -> Result<WorkerSelectionResult, KvSchedulerError> {
        self.select_worker(input)
    }
}

/// Inputs supplied by the selector's host.
#[derive(Clone, Copy)]
pub enum WorkerSelectionInput<'a, C: WorkerConfigLike> {
    Configured {
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    },
    Hosted {
        worker_ids: &'a [WorkerId],
        occupancy: Option<&'a dyn Fn(WorkerId) -> u64>,
    },
}

pub type ConfiguredSelectionInputs<'a, C> = (
    &'a HashMap<WorkerId, C>,
    &'a SchedulingRequest,
    RoutingEligibility<'a>,
    u32,
);

pub type HostedSelectionInputs<'a> = (&'a [WorkerId], Option<&'a dyn Fn(WorkerId) -> u64>);

impl<'a, C: WorkerConfigLike> WorkerSelectionInput<'a, C> {
    pub fn configured(
        workers: &'a HashMap<WorkerId, C>,
        request: &'a SchedulingRequest,
        eligibility: RoutingEligibility<'a>,
        block_size: u32,
    ) -> Self {
        Self::Configured {
            workers,
            request,
            eligibility,
            block_size,
        }
    }

    pub fn hosted(
        worker_ids: &'a [WorkerId],
        occupancy: Option<&'a dyn Fn(WorkerId) -> u64>,
    ) -> Self {
        Self::Hosted {
            worker_ids,
            occupancy,
        }
    }

    pub fn into_configured(self) -> Result<ConfiguredSelectionInputs<'a, C>, KvSchedulerError> {
        match self {
            Self::Configured {
                workers,
                request,
                eligibility,
                block_size,
            } => Ok((workers, request, eligibility, block_size)),
            Self::Hosted { .. } => Err(WorkerSelectionPolicyError::failed(
                "selector requires configured worker inputs",
            )
            .into()),
        }
    }

    pub fn into_hosted(self) -> Result<HostedSelectionInputs<'a>, KvSchedulerError> {
        match self {
            Self::Hosted {
                worker_ids,
                occupancy,
            } => Ok((worker_ids, occupancy)),
            Self::Configured { .. } => Err(WorkerSelectionPolicyError::failed(
                "selector requires hosted worker inputs",
            )
            .into()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LogitWeights {
    overlap_score_credit: f64,
    overlap_score_credit_decay: f64,
    prefill_load_scale: f64,
    shared_cache_multiplier: f64,
}

/// A built-in KV-aware candidate score and the scalar inputs that produced it.
/// This remains internal to the default policy; custom policies are not required
/// to expose the same score decomposition.
#[derive(Debug, Clone, Copy)]
struct DefaultCandidateScore {
    cost: f64,
    base_cost: f64,
    preferred_taint_multiplier: f64,
    raw_prefill_blocks: f64,
    overlap_credit_blocks: f64,
    decode_cost_blocks: f64,
    active_request_cost_blocks: f64,
    device_overlap_blocks: f64,
    host_overlap_blocks: f64,
    disk_overlap_blocks: f64,
    shared_blocks_beyond: u32,
    active_prefill_tokens: usize,
    active_decode_blocks: usize,
}

/// Host-owned accounting for the DP-rank candidate universe. These fields are
/// policy-neutral and stay meaningful when a custom policy is installed.
#[derive(Debug, Clone, Copy, Default)]
struct CandidateFilterSummary {
    eligible: usize,
    not_allowed: usize,
    constraints: usize,
    overloaded: usize,
    unavailable: usize,
}

impl CandidateFilterSummary {
    fn count_worker<C: WorkerConfigLike>(
        &mut self,
        worker_id: WorkerId,
        config: &C,
        eligibility: RoutingEligibility<'_>,
    ) {
        let replicas = config.data_parallel_size() as usize;
        if !eligibility.caller_allows_worker_id(worker_id) {
            self.not_allowed += replicas;
        } else if !eligibility.is_worker_available(worker_id) {
            self.unavailable += replicas;
        } else if !eligibility.allows_worker_ignoring_overload(worker_id, config) {
            self.constraints += replicas;
        } else if eligibility.is_worker_overloaded(worker_id) {
            self.overloaded += replicas;
        } else {
            self.eligible += replicas;
        }
    }

    fn from_eligibility<C: WorkerConfigLike>(
        workers: &HashMap<WorkerId, C>,
        eligibility: RoutingEligibility<'_>,
    ) -> Self {
        let mut summary = Self::default();
        if let Some(worker) = eligibility.pinned_worker() {
            match workers.get(&worker.worker_id) {
                Some(config) => {
                    summary.count_worker(worker.worker_id, config, eligibility);
                    summary.eligible = usize::from(summary.eligible > 0);
                    summary.not_allowed = usize::from(summary.not_allowed > 0);
                    summary.constraints = usize::from(summary.constraints > 0);
                    summary.overloaded = usize::from(summary.overloaded > 0);
                    summary.unavailable = usize::from(summary.unavailable > 0);
                }
                None => summary.unavailable = 1,
            }
            return summary;
        }
        for (&worker_id, config) in workers {
            summary.count_worker(worker_id, config, eligibility);
        }
        summary
    }

    fn filtered_count(self) -> usize {
        self.not_allowed + self.constraints + self.overloaded + self.unavailable
    }
}

/// Bounded decision evidence written to a lifecycle `router.selection` span.
/// Core mode records a stable summary. Investigation mode additionally records
/// at most four built-in KV-aware candidate decompositions.
#[derive(Clone, Copy)]
pub(crate) struct RouterSelectionTelemetry<'a> {
    span: &'a tracing::Span,
    include_candidate_details: bool,
}

impl<'a> RouterSelectionTelemetry<'a> {
    pub(crate) fn new(span: &'a tracing::Span, include_candidate_details: bool) -> Self {
        Self {
            span,
            include_candidate_details,
        }
    }

    fn record_kv_aware(
        self,
        candidates: &[(WorkerWithDpRank, DefaultCandidateScore)],
        filters: CandidateFilterSummary,
        selected_worker: WorkerWithDpRank,
        selected_score: DefaultCandidateScore,
        pool_role: &'static str,
    ) {
        debug_assert!(!candidates.is_empty());
        let mut ranked = candidates.to_vec();
        ranked.sort_unstable_by(|(left_worker, left), (right_worker, right)| {
            left.cost
                .total_cmp(&right.cost)
                .then_with(|| left_worker.cmp(right_worker))
        });
        let (best_worker, best_score) = ranked[0];
        let best_margin = ranked
            .get(1)
            .map(|(_, score)| score.cost - best_score.cost)
            .unwrap_or(0.0);

        self.span
            .record("dynamo.router.candidate.count", candidates.len() as u64);
        self.record_candidate_envelope(filters);
        self.span.record("dynamo.router.algorithm.id", "kv_aware");
        self.span.record("dynamo.router.algorithm.version", "v1");
        self.span
            .record("dynamo.router.decision.schema", "kv_aware.v1");
        self.span
            .record("dynamo.router.selection.policy", "kv_aware");
        self.span.record("dynamo.router.pool.role", pool_role);
        self.span.record(
            "dynamo.router.selected.worker.id",
            selected_worker.worker_id,
        );
        self.span.record(
            "dynamo.router.selected.dp.rank",
            selected_worker.dp_rank as u64,
        );
        self.span
            .record("dynamo.router.selected.score", selected_score.cost);
        self.span
            .record("dynamo.router.best.worker.id", best_worker.worker_id);
        self.span
            .record("dynamo.router.best.dp.rank", best_worker.dp_rank as u64);
        self.span
            .record("dynamo.router.best.score", best_score.cost);
        self.span.record("dynamo.router.best.margin", best_margin);

        if self.include_candidate_details {
            const TOP_K: usize = 4;
            self.span
                .record("dynamo.router.candidates.detail_schema", "kv_aware.v1");
            let details = ranked
                .iter()
                .take(TOP_K)
                .map(|(worker, score)| {
                    format!(
                        concat!(
                            r#"{{"worker_id":{},"dp_rank":{},"score":{:.6},"base_score":{:.6},"preferred_taint_multiplier":{:.6},"raw_prefill_blocks":{:.6},"overlap_credit_blocks":{:.6},"decode_cost_blocks":{:.6},"active_request_cost_blocks":{:.6},"device_overlap_blocks":{:.6},"host_overlap_blocks":{:.6},"disk_overlap_blocks":{:.6},"shared_blocks_beyond":{},"active_prefill_tokens":{},"active_decode_blocks":{}}}"#
                        ),
                        worker.worker_id,
                        worker.dp_rank,
                        score.cost,
                        score.base_cost,
                        score.preferred_taint_multiplier,
                        score.raw_prefill_blocks,
                        score.overlap_credit_blocks,
                        score.decode_cost_blocks,
                        score.active_request_cost_blocks,
                        score.device_overlap_blocks,
                        score.host_overlap_blocks,
                        score.disk_overlap_blocks,
                        score.shared_blocks_beyond,
                        score.active_prefill_tokens,
                        score.active_decode_blocks,
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            self.span
                .record("dynamo.router.candidates.top_k", format!("[{details}]"));
        }
    }

    fn record_candidate_envelope(self, filters: CandidateFilterSummary) {
        self.span.record(
            "dynamo.router.candidate.eligible.count",
            filters.eligible as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.count",
            filters.filtered_count() as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.not_allowed",
            filters.not_allowed as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.constraints",
            filters.constraints as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.overloaded",
            filters.overloaded as u64,
        );
        self.span.record(
            "dynamo.router.candidate.filtered.unavailable",
            filters.unavailable as u64,
        );
    }

    fn record_custom(
        self,
        candidates: &[ScoredWorkerCandidate],
        filters: CandidateFilterSummary,
        selected: ScoredWorkerCandidate,
        pool_role: &'static str,
    ) {
        self.span
            .record("dynamo.router.candidate.count", candidates.len() as u64);
        self.record_candidate_envelope(filters);
        self.span.record("dynamo.router.algorithm.id", "custom");
        self.span.record("dynamo.router.algorithm.version", "v1");
        self.span
            .record("dynamo.router.decision.schema", "selection.v1");
        self.span.record("dynamo.router.selection.policy", "custom");
        self.span.record("dynamo.router.pool.role", pool_role);
        self.span.record(
            "dynamo.router.selected.worker.id",
            selected.worker.worker_id,
        );
        self.span.record(
            "dynamo.router.selected.dp.rank",
            selected.worker.dp_rank as u64,
        );
        self.span
            .record("dynamo.router.selected.score", selected.cost);
        if let Some(best) = candidates.iter().min_by(|left, right| {
            left.cost
                .total_cmp(&right.cost)
                .then_with(|| left.worker.cmp(&right.worker))
        }) {
            self.span
                .record("dynamo.router.best.worker.id", best.worker.worker_id);
            self.span
                .record("dynamo.router.best.dp.rank", best.worker.dp_rank as u64);
            self.span.record("dynamo.router.best.score", best.cost);
            let runner_up = candidates
                .iter()
                .filter(|candidate| candidate.worker != best.worker)
                .min_by(|left, right| left.cost.total_cmp(&right.cost));
            self.span.record(
                "dynamo.router.best.margin",
                runner_up.map_or(0.0, |candidate| candidate.cost - best.cost),
            );
        }
    }
}

struct MaterializedSelectionInput<'a> {
    request: &'a SchedulingRequest,
    context: WorkerSelectionContext<'a>,
}

impl<'a> MaterializedSelectionInput<'a> {
    fn new(request: &'a SchedulingRequest, block_size: u32, weights: LogitWeights) -> Self {
        Self {
            request,
            context: WorkerSelectionContext {
                request,
                request_id: request.mode.request_id().unwrap_or("-"),
                request_blocks: request.request_blocks(block_size),
                block_size,
                track_prefill_tokens: request.track_prefill_tokens,
                weights,
                router_temperature_override: request
                    .router_config_override
                    .as_ref()
                    .and_then(|config| config.router_temperature),
            },
        }
    }

    fn row(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
    ) -> WorkerCandidate {
        self.row_with_device_overlap(
            worker,
            preferred_taint_multiplier,
            inputs,
            |_, device_overlap_blocks| device_overlap_blocks,
        )
    }

    fn row_with_device_overlap(
        &self,
        worker: WorkerWithDpRank,
        preferred_taint_multiplier: Option<f64>,
        inputs: WorkerInputs,
        select_device_overlap: impl FnOnce(f64, f64) -> f64,
    ) -> WorkerCandidate {
        let cached_tokens = if inputs.contains(WorkerInputs::CACHE)
            || (inputs.contains(WorkerInputs::LOAD) && self.request.track_prefill_tokens)
        {
            self.request.effective_cached_tokens_for(worker)
        } else {
            0
        };
        let worker_load = if inputs.contains(WorkerInputs::LOAD) {
            self.request.worker_loads.get(&worker).copied()
        } else {
            None
        };
        let cache = if inputs.contains(WorkerInputs::CACHE) {
            let effective_overlap_blocks = self.request.effective_overlap_blocks_for(worker);
            let reported_device_overlap_blocks = self
                .request
                .overlap
                .tier_overlap_blocks
                .device
                .get(&worker)
                .copied()
                .map(|blocks| blocks as f64)
                .unwrap_or(0.0);
            let device_overlap_blocks =
                select_device_overlap(effective_overlap_blocks, reported_device_overlap_blocks);
            let shared_beyond = |device_blocks: f64| {
                self.request.shared_cache_hits.as_ref().map_or(0, |hits| {
                    // `hits_beyond` expects the unweighted device prefix depth.
                    hits.hits_beyond(device_blocks.round().max(0.0) as u32)
                })
            };
            WorkerCacheInput {
                effective_overlap_blocks,
                device_overlap_blocks,
                host_overlap_blocks: self
                    .request
                    .overlap
                    .tier_overlap_blocks
                    .host_pinned
                    .get(&worker)
                    .copied()
                    .unwrap_or(0) as f64,
                disk_overlap_blocks: self
                    .request
                    .overlap
                    .tier_overlap_blocks
                    .disk
                    .get(&worker)
                    .copied()
                    .unwrap_or(0) as f64,
                shared_beyond_device_blocks: shared_beyond(device_overlap_blocks),
            }
        } else {
            WorkerCacheInput::default()
        };
        let load = if inputs.contains(WorkerInputs::LOAD) {
            let raw_prefill_tokens = if self.request.track_prefill_tokens {
                match worker_load {
                    Some(load) => {
                        // Preserve the legacy operation order when overlap exceeds the prompt.
                        let uncached_tokens = super::prefill_load::effective_prefill_tokens(
                            self.request.isl_tokens,
                            cached_tokens,
                        );
                        let projected_tokens = load.active_prefill_tokens + uncached_tokens;
                        projected_tokens.saturating_add(cached_tokens)
                    }
                    None => self.request.isl_tokens,
                }
            } else {
                0
            } as f64;
            let worker_load = worker_load.unwrap_or_default();
            WorkerLoadInput {
                raw_prefill_blocks: raw_prefill_tokens / self.context.block_size as f64,
                active_prefill_tokens: worker_load.active_prefill_tokens,
                decode_cost_blocks: worker_load.potential_decode_blocks() as f64,
                active_requests: worker_load.active_requests,
            }
        } else {
            WorkerLoadInput::default()
        };

        WorkerCandidate {
            worker,
            inputs,
            cache,
            load,
            preferred_taint_multiplier,
        }
    }
}

fn selection_result(
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    block_size: u32,
) -> WorkerSelectionResult {
    WorkerSelectionResult {
        worker,
        required_blocks: request.request_blocks(block_size),
        effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
        cached_tokens: request.effective_cached_tokens_for(worker),
        potential_decode_blocks: request
            .potential_decode_blocks_after_admission(worker, block_size),
    }
}

fn log_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    worker: WorkerWithDpRank,
    worker_type: &'static str,
    cost: f64,
    effective_overlap_blocks: f64,
) {
    let request_id = request.mode.request_id().unwrap_or("-");
    let host_pinned_blocks = request
        .overlap
        .tier_overlap_blocks
        .host_pinned
        .get(&worker)
        .copied()
        .unwrap_or(0);
    let disk_blocks = request
        .overlap
        .tier_overlap_blocks
        .disk
        .get(&worker)
        .copied()
        .unwrap_or(0);

    if request.pinned_worker == Some(worker) {
        tracing::info!(
            request_id,
            "Selected pinned worker: worker_type={}, worker_id={} dp_rank={:?}, logit: {:.3}, effective cached blocks: {:.2}",
            worker_type,
            worker.worker_id,
            worker.dp_rank,
            cost,
            effective_overlap_blocks,
        );
    } else if worker_type == "decode" {
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit = cost,
            host_pinned_blocks,
            disk_blocks,
            "Selected worker"
        );
    } else {
        let total_kv_blocks = workers
            .get(&worker.worker_id)
            .and_then(WorkerConfigLike::total_kv_blocks);
        tracing::info!(
            router_mode = "kv",
            request_id,
            worker_id = worker.worker_id,
            worker_type = %worker_type,
            dp_rank = ?worker.dp_rank,
            logit = cost,
            effective_cached_blocks = effective_overlap_blocks,
            host_pinned_blocks,
            disk_blocks,
            total_kv_blocks = ?total_kv_blocks,
            "Selected worker"
        );
    }
}

#[inline(always)]
// DefaultWorkerSelector and SelectionService both converge here. Only the scorer/picker stage is
// dispatched; eligibility outcomes and result construction stay host-owned and shared.
fn select_worker_with_policy<C: WorkerConfigLike>(
    kv_router_config: &KvRouterConfig,
    worker_type: &'static str,
    state: WorkerSelectionPolicyStateRef<'_>,
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    block_size: u32,
    telemetry: Option<RouterSelectionTelemetry<'_>>,
) -> Result<WorkerSelectionResult, KvSchedulerError> {
    assert!(request.isl_tokens > 0);
    eligibility.validate_pinned_worker_allowed()?;

    if let Some(worker) = eligibility.pinned_worker() {
        match eligibility.validate_worker_rank(workers, worker) {
            Ok(_) => {}
            Err(WorkerEligibilityError::WorkerOverloaded { .. }) => {
                return Err(KvSchedulerError::PinnedWorkerOverloaded {
                    worker_id: worker.worker_id,
                });
            }
            Err(_) => return Err(KvSchedulerError::NoEndpoints),
        }
    }

    let weights = selection_weights(kv_router_config, request);
    let input = MaterializedSelectionInput::new(request, block_size, weights);
    let filter_summary =
        telemetry.map(|_| CandidateFilterSummary::from_eligibility(workers, eligibility));
    let selected = match state {
        WorkerSelectionPolicyStateRef::Default(picker) => {
            let scorer = DefaultWorkerScorer {
                kv_router_config,
                worker_type,
            };
            pick_default_worker(
                &scorer,
                picker,
                &input,
                workers,
                request,
                eligibility,
                telemetry,
                filter_summary,
            )
        }
        WorkerSelectionPolicyStateRef::Custom(state) => {
            let mut state = state.borrow_mut();
            let has_eligible_worker =
                collect_custom_candidates(&mut state, &input, workers, request, eligibility)?;
            let CustomWorkerSelectionState {
                picker,
                picker_inputs,
                candidates,
                cache_inputs,
                load_inputs,
                ..
            } = &mut *state;
            if candidates.is_empty() {
                if has_eligible_worker {
                    return Err(KvSchedulerError::AllEligibleWorkersFiltered);
                }
                None
            } else {
                debug_assert!(
                    !picker_inputs.contains(WorkerInputs::CACHE)
                        || cache_inputs.len() == candidates.len()
                );
                debug_assert!(
                    !picker_inputs.contains(WorkerInputs::LOAD)
                        || load_inputs.len() == candidates.len()
                );
                let picker_input = WorkerInputView {
                    candidates,
                    cache: picker_inputs
                        .contains(WorkerInputs::CACHE)
                        .then_some(cache_inputs.as_slice()),
                    load: picker_inputs
                        .contains(WorkerInputs::LOAD)
                        .then_some(load_inputs.as_slice()),
                };
                let row = picker.pick(&input.context, picker_input)?;
                let Some(candidate) = candidates.get(row) else {
                    return Err(WorkerSelectionPolicyError::InvalidPickerRow {
                        row,
                        candidate_count: candidates.len(),
                    }
                    .into());
                };
                if let Some(telemetry) = telemetry {
                    telemetry.record_custom(
                        candidates,
                        filter_summary.expect("telemetry includes candidate accounting"),
                        *candidate,
                        worker_type,
                    );
                }
                Some((candidate.worker, candidate.cost))
            }
        }
    };
    let Some((worker, cost)) = selected else {
        if eligibility.has_eligible_worker_ignoring_overload(
            workers
                .iter()
                .map(|(&worker_id, config)| (worker_id, config)),
        ) {
            return Err(KvSchedulerError::AllEligibleWorkersOverloaded);
        }
        return Err(KvSchedulerError::NoEndpoints);
    };
    let result = selection_result(request, worker, block_size);
    log_selection(
        workers,
        request,
        worker,
        worker_type,
        cost,
        result.effective_overlap_blocks,
    );
    Ok(result)
}

#[cfg(test)]
mod test_support {
    use std::collections::{HashMap, HashSet};

    use rustc_hash::FxHashMap;

    use super::*;
    use crate::scheduling::{OverlapSignals, ScheduleMode};

    #[derive(Clone, Default)]
    pub(super) struct TaintedWorkerConfig {
        pub(super) taints: HashSet<String>,
    }

    impl WorkerConfigLike for TaintedWorkerConfig {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            1
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            None
        }

        fn taints(&self) -> &HashSet<String> {
            &self.taints
        }
    }

    pub(super) fn base_request(isl_tokens: usize) -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly {
                request_id: Some("test".into()),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::default(),
                effective_cached_tokens: HashMap::default(),
            },
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: None,
        }
    }

    pub(super) fn worker_loads_with_active_decode(
        decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    ) -> FxHashMap<WorkerWithDpRank, crate::sequences::WorkerLoadProjection> {
        decode_blocks
            .into_iter()
            .map(|(worker, active_decode_blocks)| {
                (
                    worker,
                    crate::sequences::WorkerLoadProjection {
                        active_decode_blocks,
                        ..Default::default()
                    },
                )
            })
            .collect()
    }
}
