# KV router routing-decision analysis scripts

Parse `dynamo_kv_router::scheduling::selector` "Routing decision" log lines from
frontend logs (`logs/frontend-*.log` in a dynamo repro run directory) and plot
routing behavior: chosen logit over time, per-candidate stacked logit
component breakdown (base cost / queued load / KV cache-hit credit), and
cross-frontend routing collisions (two frontends independently picking the
same worker within a short window, due to eventually-consistent load sync).

## Usage

Point `RUN_DIR` at a run directory containing `logs/frontend-*.log`
(defaults to `/home/sbak/dynamo_repro/runs/20260708_213527` if unset):

```bash
RUN_DIR=/home/sbak/dynamo_repro/runs/<run> python3 plot_routing.py
RUN_DIR=/home/sbak/dynamo_repro/runs/<run> python3 plot_stacked_logit.py
RUN_DIR=/home/sbak/dynamo_repro/runs/<run> python3 plot_cross_frontend.py
```

Requires matplotlib + numpy (no pandas). Output PNGs are written to
`$RUN_DIR/analysis/`.

- `plot_routing.py` -> `1_chosen_logit_over_time.png`, `2_logit_components_*.png`,
  `3_chosen_worker_*.png`, `4_candidate_spread_*.png`
- `plot_stacked_logit.py` -> `5_stacked_logit_grid_*.png`,
  `6_stacked_logit_timeline_*.png` (also supports
  `python3 plot_stacked_logit.py --request <request_id> [prefill|decode]`
  for a single-decision detail plot)
- `plot_cross_frontend.py` -> `7_cross_frontend_timeline_*.png`,
  `8_cross_frontend_summary_*.png`, `9_cross_frontend_examples_*.png`
  (imports from `plot_stacked_logit.py`, must be run from this directory)

## Logit formula (reverse-engineered from log data, verified against traces)

```
logit = prefill_load_scale * max(raw_prefill_blocks - overlap_credit_blocks, 0) + decode_cost_blocks
```

- `worker_type=prefill`: `raw_prefill_blocks` = (this request's isl_tokens +
  that candidate's already-queued prefill tokens) / block_size (16).
  `overlap_credit_blocks` = that candidate's KV cache-hit blocks (device
  overlap), subtracted from the combined total before the max(...,0) floor.
- `worker_type=decode`: `raw_prefill_blocks` is always 0 (no cache-affinity
  term); logit = this request's own estimated decode cost + that candidate's
  already-queued decode blocks. Decode routing explicitly zeroes
  `overlap_score_credit` (confirmed in source, `prefill_router/mod.rs:404`).
