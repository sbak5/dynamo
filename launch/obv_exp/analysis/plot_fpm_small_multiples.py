#!/usr/bin/env python3
"""Render readable per-worker FPM load timelines as small multiples."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_DIR = Path(__file__).resolve().parent
BLOCK_SIZE = 16
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
PREFILL_WARMUP_S = float(os.environ.get("PREFILL_WARMUP_S", "15"))


def load(role: str) -> dict[str, list[tuple[datetime, float, float]]]:
    result: dict[str, list[tuple[datetime, float, float]]] = defaultdict(list)
    path = RUN_DIR / "logs" / f"fpm-{role}.jsonl"
    with path.open() as stream:
        for line in stream:
            raw = json.loads(line)
            metrics = raw["metrics"]
            queued = metrics["queued_requests"]
            scheduled = metrics["scheduled_requests"]
            if role == "prefill":
                q = float(queued["sum_prefill_tokens"])
                s = float(scheduled["sum_prefill_tokens"])
            else:
                q = float(queued["sum_decode_kv_tokens"]) / BLOCK_SIZE
                s = float(scheduled["sum_decode_kv_tokens"]) / BLOCK_SIZE
            result[str(metrics["worker_id"])[-5:]].append(
                (datetime.fromisoformat(raw["received_at"]), q, q + s)
            )
    return dict(sorted(result.items()))


def plot(role: str) -> None:
    records = load(role)
    initial_t0 = min(row[0] for rows in records.values() for row in rows)
    warmup_s = PREFILL_WARMUP_S if role == "prefill" else 0.0
    if warmup_s:
        records = {
            worker: [
                row
                for row in rows
                if (row[0] - initial_t0).total_seconds() >= warmup_s
            ]
            for worker, rows in records.items()
        }
    t0 = min(row[0] for rows in records.values() for row in rows)
    unit = "tokens" if role == "prefill" else "16-token blocks"
    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1, 1.1]})
    max_load = max(row[2] for rows in records.values() for row in rows)
    all_series = []
    for axis, ((worker, rows), color) in zip(axes[:4], zip(records.items(), COLORS)):
        times = [(row[0] - t0).total_seconds() for row in rows]
        queued = [row[1] for row in rows]
        total = [row[2] for row in rows]
        axis.step(times, total, where="post", color=color, linewidth=1.25, label="queued + scheduled")
        axis.step(times, queued, where="post", color=color, linewidth=0.9, linestyle=":", label="queued only")
        axis.set_ylim(0, max_load * 1.04)
        axis.set_ylabel(f"worker {worker}\n{unit}")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", fontsize=8)
        all_series.append((times, total))

    # Values are sampled on the shared FPM receive cadence; draw balance envelope.
    bucket: dict[float, list[float]] = defaultdict(list)
    for times, values in all_series:
        for time, value in zip(times, values):
            bucket[round(time, 2)].append(value)
    times = sorted(bucket)
    means = [sum(bucket[t]) / len(bucket[t]) for t in times]
    lows = [min(bucket[t]) for t in times]
    highs = [max(bucket[t]) for t in times]
    axis = axes[4]
    axis.plot(times, means, color="#333333", linewidth=1.25, label="workers' mean total load")
    axis.fill_between(times, lows, highs, color="#999999", alpha=0.35, label="min–max worker total load")
    axis.set_ylabel(f"across workers\n{unit}")
    axis.set_xlabel("time (s, relative to first retained FPM event)")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Observed FPM load by {role} worker"
        + (f" — initial {warmup_s:.0f}s excluded" if warmup_s else ""),
        y=0.995,
    )
    fig.tight_layout()
    output = RUN_DIR / "analysis" / f"fpm_{role}_small_multiples.png"
    fig.savefig(output, dpi=170)
    print(f"wrote {output}")


for role in ("prefill", "decode"):
    plot(role)
