#!/usr/bin/env python3
"""Compare router-observed candidate load with engine FPM load snapshots."""

from __future__ import annotations

import bisect
import csv
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_routing import parse_file


RUN_DIR = os.environ.get("RUN_DIR", ".")
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "16"))
MAX_SAMPLE_AGE_S = float(os.environ.get("MAX_FPM_SAMPLE_AGE_S", "2.0"))
# The initial admission burst is useful for debugging startup, but it can be tens
# of thousands of tokens and obscure the sustained prefill queue.  Keep it out
# of the steady-state comparison plot while retaining every record in the CSV.
PREFILL_WARMUP_S = float(os.environ.get("PREFILL_WARMUP_S", "15"))
PREFILL_DISPLAY_BIN_S = float(os.environ.get("PREFILL_DISPLAY_BIN_S", "2"))


def load_router_records() -> list[dict]:
    records: list[dict] = []
    for path in sorted(glob.glob(f"{RUN_DIR}/logs/frontend-*.log")):
        source = os.path.basename(path).removesuffix(".log")
        for record in parse_file(path):
            record["source"] = source
            records.append(record)
    return sorted(records, key=lambda record: record["ts"])


def load_fpm(role: str) -> list[dict]:
    path = f"{RUN_DIR}/logs/fpm-{role}.jsonl"
    records: list[dict] = []
    with open(path, errors="replace") as stream:
        for line in stream:
            try:
                raw = json.loads(line)
                metrics = raw["metrics"]
                timestamp = datetime.fromisoformat(raw["received_at"])
                queued = metrics["queued_requests"]
                scheduled = metrics["scheduled_requests"]
                if role == "prefill":
                    queued_load = float(queued["sum_prefill_tokens"])
                    scheduled_load = float(scheduled["sum_prefill_tokens"])
                    unit = "tokens"
                else:
                    queued_load = float(queued["sum_decode_kv_tokens"]) / BLOCK_SIZE
                    scheduled_load = (
                        float(scheduled["sum_decode_kv_tokens"]) / BLOCK_SIZE
                    )
                    unit = f"{BLOCK_SIZE}-token blocks"
                records.append(
                    {
                        "ts": timestamp,
                        "worker_id": str(metrics["worker_id"]),
                        "dp_rank": int(metrics["dp_rank"]),
                        "counter_id": int(metrics["counter_id"]),
                        "queued_load": queued_load,
                        "scheduled_load": scheduled_load,
                        "engine_load": queued_load + scheduled_load,
                        "unit": unit,
                    }
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    return sorted(records, key=lambda record: record["ts"])


def group_fpm(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["worker_id"]].append(record)
    return dict(grouped)


def resolve_worker(short_id: str, worker_ids: list[str]) -> str | None:
    matches = [worker_id for worker_id in worker_ids if worker_id.endswith(short_id)]
    return matches[0] if len(matches) == 1 else None


def latest_snapshot(records: list[dict], timestamp: datetime) -> tuple[dict | None, float]:
    timestamps = [record["ts"] for record in records]
    index = bisect.bisect_right(timestamps, timestamp) - 1
    if index < 0:
        return None, math.inf
    record = records[index]
    age = (timestamp - record["ts"]).total_seconds()
    if age > MAX_SAMPLE_AGE_S:
        return None, age
    return record, age


def build_joined_rows(
    router_records: list[dict], role: str, fpm_by_worker: dict[str, list[dict]]
) -> list[dict]:
    rows: list[dict] = []
    worker_ids = sorted(fpm_by_worker)
    for decision in router_records:
        if decision["worker_type"] != role:
            continue
        for candidate in decision["candidates"]:
            full_worker_id = resolve_worker(candidate["worker_id"], worker_ids)
            if full_worker_id is None:
                continue
            snapshot, age = latest_snapshot(fpm_by_worker[full_worker_id], decision["ts"])
            if snapshot is None:
                continue
            router_load = float(
                candidate["prefill_tok"]
                if role == "prefill"
                else candidate["decode_blk"]
            )
            # Candidate scores are expressed in 16-token blocks. Preserve the
            # router's cache-overlap contribution and replace only this active
            # load term with the worker-observed FPM load below.
            router_active_load_blocks = (
                float(candidate["prefill_tok"]) / BLOCK_SIZE
                + float(candidate["decode_blk"])
            )
            engine_total_blocks = (
                snapshot["engine_load"] / BLOCK_SIZE
                if role == "prefill"
                else snapshot["engine_load"]
            )
            chosen = str(decision["chosen_worker_id"]) == full_worker_id
            rows.append(
                {
                    "timestamp": decision["ts"].isoformat(),
                    "request_id": decision["request_id"],
                    "frontend": decision["source"],
                    "role": role,
                    "worker_id": full_worker_id,
                    "dp_rank": candidate["dp_rank"],
                    "chosen": chosen,
                    "candidate_local_logit": candidate["logit"],
                    "candidate_overlap_blocks": candidate["overlap"],
                    "router_active_load_blocks": router_active_load_blocks,
                    "router_load": router_load,
                    "engine_queued_load": snapshot["queued_load"],
                    "engine_scheduled_load": snapshot["scheduled_load"],
                    "engine_total_load": snapshot["engine_load"],
                    "engine_total_blocks": engine_total_blocks,
                    "discrepancy": router_load - snapshot["engine_load"],
                    "sample_age_ms": age * 1000.0,
                    "fpm_counter_id": snapshot["counter_id"],
                }
            )
    return rows


def write_rows(rows: list[dict], role: str, output_dir: str) -> None:
    path = f"{output_dir}/router_vs_fpm_{role}.csv"
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_engine_observed_regret(rows: list[dict]) -> list[dict]:
    """Compare the selected worker with the lowest FPM-observed candidate load.

    This is intentionally measured in queued+scheduled work, not milliseconds:
    aggregate FPM has no request-level scheduler-admission timestamp, so it cannot
    establish a counterfactual start time for a worker that did not receive a
    request.
    """
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["request_id"]:
            by_request[str(row["request_id"])].append(row)

    result: list[dict] = []
    for request_id, candidates in by_request.items():
        chosen = next((row for row in candidates if row["chosen"]), None)
        if chosen is None or len(candidates) < 2:
            continue
        best = min(candidates, key=lambda row: row["engine_total_load"])
        chosen_load = chosen["engine_total_load"]
        best_load = best["engine_total_load"]
        result.append(
            {
                "timestamp": chosen["timestamp"],
                "request_id": request_id,
                "frontend": chosen["frontend"],
                "chosen_worker_id": chosen["worker_id"],
                "best_worker_id": best["worker_id"],
                "candidate_count_with_fpm": len(candidates),
                "chosen_engine_load": chosen_load,
                "best_engine_load": best_load,
                "avoidable_engine_load": chosen_load - best_load,
                "chosen_engine_rank": 1
                + sum(
                    candidate["engine_total_load"] < chosen_load
                    for candidate in candidates
                ),
                "chosen_fpm_sample_age_ms": chosen["sample_age_ms"],
                "best_fpm_sample_age_ms": best["sample_age_ms"],
            }
        )
    return sorted(result, key=lambda row: row["timestamp"])


def write_and_plot_engine_observed_regret(
    rows: list[dict], role: str, output_dir: str
) -> None:
    regret_rows = build_engine_observed_regret(rows)
    if not regret_rows:
        return
    csv_path = f"{output_dir}/engine_observed_regret_{role}.csv"
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regret_rows[0]))
        writer.writeheader()
        writer.writerows(regret_rows)

    t0 = datetime.fromisoformat(regret_rows[0]["timestamp"])
    times = [
        (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
        for row in regret_rows
    ]
    regrets = [row["avoidable_engine_load"] for row in regret_rows]
    nonzero = sorted(value for value in regrets if value > 0)
    unit = "tokens" if role == "prefill" else f"{BLOCK_SIZE}-token blocks"
    nonbest = sum(value > 0 for value in regrets)

    fig, (timeline_ax, cdf_ax) = plt.subplots(2, 1, figsize=(13, 8))
    timeline_ax.scatter(times, regrets, s=15, alpha=0.55, color="#d55e00")
    timeline_ax.axhline(0, color="black", linewidth=0.8)
    timeline_ax.set_xlabel("time (s, relative to first joined decision)")
    timeline_ax.set_ylabel(f"chosen − best candidate ({unit})")
    timeline_ax.set_title("Per-decision avoidable engine-observed backlog")
    timeline_ax.grid(alpha=0.25)

    if nonzero:
        cdf_ax.plot(
            nonzero,
            [(index + 1) / len(nonzero) for index in range(len(nonzero))],
            color="#0072b2",
            linewidth=1.6,
        )
    cdf_ax.set_xlabel(f"avoidable engine-observed backlog ({unit})")
    cdf_ax.set_ylabel("CDF among non-best decisions")
    cdf_ax.set_title(
        f"{nonbest}/{len(regret_rows)} decisions selected a candidate above the "
        "minimum FPM-observed load"
    )
    cdf_ax.grid(alpha=0.25)
    fig.suptitle(
        "Counterfactual routing opportunity (work units, not estimated time)\n"
        "FPM is aggregate; request-level scheduler-start tracing is required for ms savings.",
        fontsize=11,
    )
    fig.tight_layout()
    path = f"{output_dir}/11_engine_observed_regret_{role}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(
        f"wrote {path} ({nonbest}/{len(regret_rows)} non-best by engine-observed load)"
    )


def build_load_rebased_regret(rows: list[dict]) -> list[dict]:
    """Replay candidate ranking with cache score fixed and FPM load substituted.

    `candidate_local_logit` is the router's actual score. Subtracting the
    candidate's locally tracked active-load term and adding the FPM-observed
    active-load term preserves the radix-index/overlap contribution. This is
    an offline load-rebased score, not a request-level counterfactual latency.
    """
    by_request: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["request_id"]:
            by_request[str(row["request_id"])].append(row)

    result: list[dict] = []
    for request_id, candidates in by_request.items():
        chosen = next((row for row in candidates if row["chosen"]), None)
        if chosen is None or len(candidates) < 2:
            continue
        for candidate in candidates:
            candidate["load_rebased_logit"] = (
                candidate["candidate_local_logit"]
                - candidate["router_active_load_blocks"]
                + candidate["engine_total_blocks"]
            )
        rebased_best = min(candidates, key=lambda row: row["load_rebased_logit"])
        local_best = min(candidates, key=lambda row: row["candidate_local_logit"])
        result.append(
            {
                "timestamp": chosen["timestamp"],
                "request_id": request_id,
                "frontend": chosen["frontend"],
                "chosen_worker_id": chosen["worker_id"],
                "rebased_best_worker_id": rebased_best["worker_id"],
                "candidate_count_with_fpm": len(candidates),
                "chosen_local_logit": chosen["candidate_local_logit"],
                "chosen_rebased_logit": chosen["load_rebased_logit"],
                "rebased_best_logit": rebased_best["load_rebased_logit"],
                "avoidable_rebased_logit": (
                    chosen["load_rebased_logit"] - rebased_best["load_rebased_logit"]
                ),
                "chosen_is_rebased_best": chosen["worker_id"] == rebased_best["worker_id"],
                "chosen_fpm_sample_age_ms": chosen["sample_age_ms"],
                "best_fpm_sample_age_ms": rebased_best["sample_age_ms"],
            }
        )
    return sorted(result, key=lambda row: row["timestamp"])


def write_and_plot_load_rebased_regret(
    rows: list[dict], role: str, output_dir: str
) -> None:
    regret_rows = build_load_rebased_regret(rows)
    if not regret_rows:
        return
    csv_path = f"{output_dir}/load_rebased_logit_regret_{role}.csv"
    with open(csv_path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regret_rows[0]))
        writer.writeheader()
        writer.writerows(regret_rows)

    t0 = datetime.fromisoformat(regret_rows[0]["timestamp"])
    times = [
        (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
        for row in regret_rows
    ]
    regrets = [row["avoidable_rebased_logit"] for row in regret_rows]
    nonzero = sorted(value for value in regrets if value > 0)
    changed = sum(value > 0 for value in regrets)
    fig, (timeline_ax, cdf_ax) = plt.subplots(2, 1, figsize=(13, 8))
    timeline_ax.scatter(times, regrets, s=15, alpha=0.55, color="#cc79a7")
    timeline_ax.axhline(0, color="black", linewidth=0.8)
    timeline_ax.set_xlabel("time (s, relative to first joined decision)")
    timeline_ax.set_ylabel("chosen − rebased-best logit (blocks)")
    timeline_ax.set_title("Per-decision load-rebased logit regret")
    timeline_ax.grid(alpha=0.25)
    if nonzero:
        cdf_ax.plot(
            nonzero,
            [(index + 1) / len(nonzero) for index in range(len(nonzero))],
            color="#0072b2",
            linewidth=1.6,
        )
    cdf_ax.set_xlabel("avoidable load-rebased logit (blocks)")
    cdf_ax.set_ylabel("CDF among selections changed by FPM load")
    cdf_ax.set_title(
        f"{changed}/{len(regret_rows)} local choices differ from the FPM-load-rebased best"
    )
    cdf_ax.grid(alpha=0.25)
    fig.suptitle(
        "Cache/overlap term retained from router; only active-load term replaced by FPM",
        fontsize=11,
    )
    fig.tight_layout()
    path = f"{output_dir}/12_load_rebased_logit_regret_{role}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path} ({changed}/{len(regret_rows)} changed selections)")


def display_series(records: list[dict], t0: datetime, bin_s: float) -> tuple[list[float], list[float], list[float]]:
    """Mean FPM loads in display bins, avoiding dense step-transition ink."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        bucket = int((record["ts"] - t0).total_seconds() // bin_s)
        buckets[bucket].append(record)
    times, queued, total = [], [], []
    for bucket, samples in sorted(buckets.items()):
        times.append((bucket + 0.5) * bin_s)
        queued.append(sum(sample["queued_load"] for sample in samples) / len(samples))
        total.append(sum(sample["engine_load"] for sample in samples) / len(samples))
    return times, queued, total


def plot_role(
    router_records: list[dict],
    role: str,
    fpm_by_worker: dict[str, list[dict]],
    rows: list[dict],
    output_dir: str,
) -> None:
    if not fpm_by_worker:
        print(f"No {role} FPM records found")
        return
    all_times = [record["ts"] for records in fpm_by_worker.values() for record in records]
    t0 = min(all_times)
    warmup_s = PREFILL_WARMUP_S if role == "prefill" else 0.0
    if warmup_s:
        fpm_by_worker = {
            worker_id: [
                record
                for record in records
                if (record["ts"] - t0).total_seconds() >= warmup_s
            ]
            for worker_id, records in fpm_by_worker.items()
        }
        rows = [
            row
            for row in rows
            if (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
            >= warmup_s
        ]
    unit = next(iter(fpm_by_worker.values()))[0]["unit"]
    colors = plt.get_cmap("tab10")
    worker_ids = sorted(fpm_by_worker)
    worker_colors = {
        worker_id: colors(index % 10) for index, worker_id in enumerate(worker_ids)
    }
    worker_positions = {worker_id: index for index, worker_id in enumerate(worker_ids)}
    fig, (load_ax, decision_ax, diff_ax) = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.2, 2.5]},
    )

    for worker_id, records in sorted(fpm_by_worker.items()):
        color = worker_colors[worker_id]
        if role == "prefill":
            times, queued_load, engine_load = display_series(
                records, t0, PREFILL_DISPLAY_BIN_S
            )
        else:
            times = [(record["ts"] - t0).total_seconds() for record in records]
            queued_load = [record["queued_load"] for record in records]
            engine_load = [record["engine_load"] for record in records]
        load_ax.plot(
            times,
            engine_load,
            color=color,
            linewidth=1.2,
            label=f"worker {worker_id[-5:]} queued+scheduled",
        )
        load_ax.plot(
            times,
            queued_load,
            color=color,
            linewidth=1.0,
            linestyle=":",
            alpha=0.8,
        )

    chosen_rows = [row for row in rows if row["chosen"]]
    frontend_markers = {"frontend-0": "o", "frontend-1": "^"}
    for frontend, marker in frontend_markers.items():
        for worker_id in worker_ids:
            selected = [
                row
                for row in chosen_rows
                if row["frontend"] == frontend and row["worker_id"] == worker_id
            ]
            if not selected:
                continue
            times = [
                (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
                for row in selected
            ]
            color = worker_colors[worker_id]
            load_ax.scatter(
                times,
                [row["engine_total_load"] for row in selected],
                marker=marker,
                s=38,
                color=color,
                edgecolors="black",
                linewidths=0.55,
                zorder=5,
            )
            decision_ax.scatter(
                times,
                [worker_positions[worker_id]] * len(selected),
                marker=marker,
                s=36,
                color=color,
                edgecolors="black",
                linewidths=0.55,
                zorder=5,
            )

    worker_handles = [
        Line2D(
            [0],
            [0],
            color=worker_colors[worker_id],
            linewidth=1.8,
            label=f"worker {worker_id[-5:]}",
        )
        for worker_id in worker_ids
    ]
    frontend_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=marker,
            markerfacecolor="white",
            linewidth=0,
            label=f"{frontend} decision",
        )
        for frontend, marker in frontend_markers.items()
    ]

    unchosen_rows = [row for row in rows if not row["chosen"]]
    diff_ax.scatter(
        [
            (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
            for row in unchosen_rows
        ],
        [row["discrepancy"] for row in unchosen_rows],
        s=13,
        alpha=0.35,
        color="#999999",
        label="unchosen candidates",
    )
    for worker_id in worker_ids:
        selected = [
            row for row in chosen_rows if row["worker_id"] == worker_id
        ]
        if not selected:
            continue
        diff_ax.scatter(
            [
                (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()
                for row in selected
            ],
            [row["discrepancy"] for row in selected],
            s=28,
            alpha=0.9,
            color=worker_colors[worker_id],
            edgecolors="black",
            linewidths=0.35,
            label=f"chosen worker {worker_id[-5:]}",
        )

    load_ax.set_ylabel(f"engine-observed load ({unit})")
    load_ax.set_title(
        f"Router decisions over engine load timeline — {role}\n"
        "solid = queued + scheduled, dotted = queued only; decision color = chosen worker"
        + (f"; startup first {warmup_s:.0f}s excluded" if warmup_s else "")
        + (f"; FPM display averaged in {PREFILL_DISPLAY_BIN_S:g}s bins" if role == "prefill" else "")
    )
    load_ax.grid(alpha=0.25)
    load_ax.legend(handles=worker_handles + frontend_handles, fontsize=8, ncol=3)
    decision_ax.set_yticks(list(worker_positions.values()))
    decision_ax.set_yticklabels([f"worker {worker_id[-5:]}" for worker_id in worker_ids])
    decision_ax.set_ylabel("selected worker")
    decision_ax.set_title(
        "Every router decision: color identifies the selected candidate; shape identifies frontend"
    )
    decision_ax.grid(axis="x", alpha=0.25)
    diff_ax.axhline(0.0, color="black", linewidth=0.8)
    diff_ax.set_xlabel("time (s, relative to first FPM event)")
    diff_ax.set_ylabel(f"router load − engine queued+scheduled ({unit})")
    diff_ax.set_title("Router/engine load discrepancy at each candidate decision")
    diff_ax.grid(alpha=0.25)
    diff_ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{output_dir}/10_router_vs_fpm_{role}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path} ({len(rows)} joined candidate samples)")


def report_counter_gaps(role: str, fpm_by_worker: dict[str, list[dict]]) -> None:
    for worker_id, records in sorted(fpm_by_worker.items()):
        counters = [record["counter_id"] for record in records]
        gaps = sum(max(current - previous - 1, 0) for previous, current in zip(counters, counters[1:]))
        print(
            f"{role} worker={worker_id} events={len(records)} "
            f"counter_gaps={gaps}"
        )


def main() -> None:
    router_records = load_router_records()
    output_dir = f"{RUN_DIR}/analysis"
    os.makedirs(output_dir, exist_ok=True)
    print(f"router decisions={len(router_records)}")
    for role in ("prefill", "decode"):
        fpm_by_worker = group_fpm(load_fpm(role))
        report_counter_gaps(role, fpm_by_worker)
        rows = build_joined_rows(router_records, role, fpm_by_worker)
        write_rows(rows, role, output_dir)
        write_and_plot_engine_observed_regret(rows, role, output_dir)
        write_and_plot_load_rebased_regret(rows, role, output_dir)
        plot_role(router_records, role, fpm_by_worker, rows, output_dir)


if __name__ == "__main__":
    main()
