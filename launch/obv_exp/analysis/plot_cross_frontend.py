#!/usr/bin/env python3
"""Cross-frontend routing collision analysis.

Two frontends (frontend-0, frontend-1) each run their own KV router
instance against the SAME shared pool of 4 prefill + 4 decode workers
(confirmed: identical worker_id sets logged in each frontend's startup
"Adding worker" lines). Each router's logit computation is a purely LOCAL
view: it only reflects requests *that frontend itself* has already
dispatched (queued_prefill/queued_decode/etc in the routing-decision log
come from that frontend's own KV-event/discovery state).

If frontend-0 and frontend-1 independently decide, within a short window,
that the SAME worker is cheapest -- each based on a snapshot that doesn't
yet reflect the other's about-to-land request -- they both dispatch there.
The worker then receives ~2x the load either frontend's local cost
function accounted for. This is the classic distributed/multi-scheduler
"thundering herd on stale state" problem: locally optimal != globally
optimal when the cost function's inputs lag reality.

This script finds such collisions (same chosen worker, different
frontends, within `--window` seconds) and visualizes:
  1. A timeline scatter of which worker each frontend chose over time,
     with collisions circled.
  2. Side-by-side "local view at decision time" stacked bars for a sample
     of collisions, using the same rendering as plot_stacked_logit.py, to
     show both frontends saw the same worker as cheap independently.
  3. A per-worker collision-count summary.
"""

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_stacked_logit import (
    parse_decisions, parse_weights, compute_components,
    draw_candidate_bar, legend_handles,
)

RUN_DIR = os.environ.get("RUN_DIR", "/home/sbak/dynamo_repro/runs/20260708_213527")
LOG_DIR = f"{RUN_DIR}/logs"
OUT_DIR = f"{RUN_DIR}/analysis"
COLLISION_WINDOW_S = 0.3


def load_all():
    weights = {}
    per_source = {}
    for path in sorted(glob.glob(f"{LOG_DIR}/frontend-*.log")):
        source = path.split("/")[-1].replace(".log", "")
        weights.update(parse_weights(path))
        recs = parse_decisions(path)
        for r in recs:
            r["source"] = source
        per_source[source] = recs

    all_records = [r for recs in per_source.values() for r in recs]
    all_records.sort(key=lambda r: r["ts"])
    t0 = all_records[0]["ts"]
    for r in all_records:
        r["t_rel"] = (r["ts"] - t0).total_seconds()
    return all_records, weights, sorted(per_source.keys())


def find_collisions(records, worker_type, window=COLLISION_WINDOW_S):
    wrecs = sorted([r for r in records if r["worker_type"] == worker_type],
                    key=lambda r: r["ts"])
    collisions = []
    for i in range(len(wrecs)):
        for j in range(i + 1, len(wrecs)):
            dt = (wrecs[j]["ts"] - wrecs[i]["ts"]).total_seconds()
            if dt > window:
                break
            if (wrecs[i]["source"] != wrecs[j]["source"]
                    and wrecs[i]["chosen_worker_id"] == wrecs[j]["chosen_worker_id"]):
                collisions.append((wrecs[i], wrecs[j], dt))
    return collisions


def plot_timeline_scatter(records, worker_type, collisions, out_path):
    wrecs = [r for r in records if r["worker_type"] == worker_type]
    worker_ids = sorted({r["chosen_worker_id"] for r in wrecs})
    w_idx = {w: i for i, w in enumerate(worker_ids)}
    sources = sorted({r["source"] for r in wrecs})
    marker = {sources[0]: "o", sources[1]: "^"} if len(sources) >= 2 else {sources[0]: "o"}
    color = {sources[0]: "#0072b2", sources[1]: "#d55e00"} if len(sources) >= 2 else {sources[0]: "#0072b2"}

    fig, ax = plt.subplots(figsize=(15, 4.5))
    for s in sources:
        srecs = [r for r in wrecs if r["source"] == s]
        ax.scatter([r["t_rel"] for r in srecs],
                   [w_idx[r["chosen_worker_id"]] for r in srecs],
                   marker=marker[s], color=color[s], s=45, alpha=0.75,
                   label=s, zorder=3, edgecolor="black", linewidth=0.3)

    for r0, r1, dt in collisions:
        y = w_idx[r0["chosen_worker_id"]]
        x = (r0["t_rel"] + r1["t_rel"]) / 2
        ax.scatter([x], [y], s=260, facecolors="none", edgecolors="red",
                   linewidths=1.8, zorder=2)

    ax.scatter([], [], s=260, facecolors="none", edgecolors="red", linewidths=1.8,
               label=f"collision (same worker, <{COLLISION_WINDOW_S}s apart, different frontends)")
    ax.set_yticks(range(len(worker_ids)))
    ax.set_yticklabels([w[-4:] for w in worker_ids])
    ax.set_xlabel("time (s, relative to first decision)")
    ax.set_ylabel("chosen worker")
    ax.set_title(f"Chosen worker over time by frontend — {worker_type} routing\n"
                 f"{len(collisions)} collisions where both frontends independently "
                 f"picked the same worker within {COLLISION_WINDOW_S}s")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path} ({len(collisions)} collisions)")


def plot_collision_examples(collisions, weights, worker_type, out_path, n=6):
    if not collisions:
        print(f"no {worker_type} collisions to plot")
        return
    # Spread samples across the timeline rather than just the first n
    idxs = sorted(set(int(round(i)) for i in
                       [k * (len(collisions) - 1) / (n - 1) for k in range(n)])) \
        if len(collisions) > 1 else [0]
    sample = [collisions[i] for i in idxs]

    fig, axes = plt.subplots(len(sample), 2, figsize=(10, 4.6 * len(sample)), squeeze=False)
    for row, (r0, r1, dt) in enumerate(sample):
        collision_note = (f"COLLISION: both chose worker {r0['chosen_worker_id'][-4:]}, "
                           f"{dt * 1000:.0f} ms apart\nneither saw the other's request")
        for col, rec in enumerate([r0, r1]):
            ax = axes[row][col]
            comps = compute_components(rec, weights)
            comps.sort(key=lambda c: c["worker_id"])
            labels = [c["worker_id"][-4:] for c in comps]
            for i, c in enumerate(comps):
                draw_candidate_bar(ax, i, 0.65, c, chosen_marker="text")
            ax.set_xticks(range(len(comps)))
            ax.set_xticklabels(labels, fontsize=8)
            title = (f"{rec['source']}  t={rec['t_rel']:.2f}s\n"
                     f"chose {rec['chosen_worker_id'][-4:]}  logit={rec['chosen_logit']:.1f}")
            if col == 0:
                title = collision_note + "\n" + title
            ax.set_title(title, fontsize=8.5,
                         color="red" if col == 0 else "black",
                         fontweight="bold" if col == 0 else "normal")
            ax.set_ylabel("logit (KV blocks)", fontsize=8)

    fig.suptitle(f"Same collision, two independent local views — {worker_type} routing\n"
                 f"left = frontend that decided first, right = frontend that decided second "
                 f"(still sees the worker as cheap -- other request hasn't landed/propagated yet)",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({len(sample)} of {len(collisions)} collisions shown)")


def plot_collision_summary(records, worker_type, collisions, out_path):
    wrecs = [r for r in records if r["worker_type"] == worker_type]
    worker_ids = sorted({r["chosen_worker_id"] for r in wrecs})
    sources = sorted({r["source"] for r in wrecs})

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(worker_ids))
    width = 0.35
    for si, s in enumerate(sources):
        counts = [sum(1 for r in wrecs if r["source"] == s and r["chosen_worker_id"] == w)
                  for w in worker_ids]
        ax.bar([xi + (si - 0.5) * width for xi in x], counts, width=width,
               label=f"chosen by {s}", alpha=0.85)
    coll_counts = [sum(1 for r0, r1, dt in collisions if r0["chosen_worker_id"] == w)
                   for w in worker_ids]
    ax.plot(x, coll_counts, "r*-", markersize=14, linewidth=1.5,
            label="collisions on that worker")
    ax.set_xticks(list(x))
    ax.set_xticklabels([w[-4:] for w in worker_ids])
    ax.set_ylabel("count")
    ax.set_title(f"Per-worker selection counts and collisions — {worker_type} routing")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    records, weights, sources = load_all()
    print(f"frontends: {sources}")

    for wt in ["prefill", "decode"]:
        collisions = find_collisions(records, wt)
        plot_timeline_scatter(records, wt, collisions, f"{OUT_DIR}/7_cross_frontend_timeline_{wt}.png")
        plot_collision_summary(records, wt, collisions, f"{OUT_DIR}/8_cross_frontend_summary_{wt}.png")
        plot_collision_examples(collisions, weights, wt, f"{OUT_DIR}/9_cross_frontend_examples_{wt}.png")


if __name__ == "__main__":
    main()
