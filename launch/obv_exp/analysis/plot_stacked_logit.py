#!/usr/bin/env python3
"""Per-request stacked-bar view of the KV router logit: one bar per candidate
worker, stacked by logit component, so the shortest bar (= chosen worker)
and its dominant component are visible at a glance.

logit = prefill_load_scale * max(raw_prefill_blocks - overlap_credit_blocks, 0)
        + decode_cost_blocks

Formula verified directly against log data (see conversation): for
worker_type=prefill, raw_prefill_blocks for a CANDIDATE (not just the chosen
worker) is the *combined* cost of this new request's own prefill
(isl_tokens/block_size, constant per decision) plus that candidate's already
queued prefill tokens from other in-flight requests (prefill_tok/block_size).
overlap_credit_blocks == device_overlap_blocks 1:1 in this trace (host/disk
hit weights don't come into play — no host/disk cache hits occurred), and it
subtracts from the COMBINED (new + queued) total before the max(...,0) floor,
confirmed against a case with both nonzero overlap and nonzero prefill_tok
simultaneously.

For worker_type=decode, raw_prefill_blocks is always 0 (credit is irrelevant
there); the candidate's logit is base_decode (this new request's own
estimated decode cost, constant per decision) + decode_blk (that candidate's
already-queued decode blocks from other requests).

Because a credit can't be attributed to "the new request's own blocks" vs
"queued blocks" individually (it subtracts from their sum), we draw the full
GROSS stack (base + queued + decode) and carve the credited amount off the
TOP as a hatched "not charged" region, with a bold line marking the actual
logit — this is honest about what's known vs. how the credit is internally
apportioned.
"""

import glob
import os
import re
import sys
from datetime import datetime

RUN_DIR = os.environ.get("RUN_DIR", "/home/sbak/dynamo_repro/runs/20260708_213527")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(r"^(?P<ts>\S+)\s+INFO.*(?:Selected worker|Routing decision) (?P<fields>.*)$")
FORMULA_RE = re.compile(r"^(?P<ts>\S+)\s+INFO.*Routing formula (?P<fields>.*)$")
KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
CAND_RE = re.compile(
    r"(\d+):(\d+):([\d.]+)\(overlap=([\d.]+),prefill_tok=(\d+),decode_blk=(\d+)\)"
)
BLOCK_SIZE = 16  # confirmed empirically: raw_prefill_blocks == isl_tokens / 16


def parse_weights(path):
    weights = {}
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            m = FORMULA_RE.match(line.strip())
            if not m:
                continue
            kv = {}
            for km in KV_RE.finditer(m.group("fields")):
                k, v = km.groups()
                kv[k] = v.strip('"')
            wt = kv.get("worker_type")
            weights[wt] = dict(
                prefill_load_scale=float(kv.get("prefill_load_scale", 1.0)),
                overlap_score_credit=float(kv.get("overlap_score_credit", 1.0)),
                overlap_score_credit_decay=float(kv.get("overlap_score_credit_decay", 0.0)),
            )
    return weights


def parse_decisions(path):
    records = []
    with open(path, errors="replace") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            if "Selected worker" not in line and "Routing decision" not in line:
                continue
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            ts = datetime.fromisoformat(m.group("ts").replace("Z", "+00:00"))
            fields_str = m.group("fields")
            cand_m = re.search(r"candidates=(\S+(?:,\S+)*)$", fields_str)
            candidates = []
            if cand_m:
                for cm in CAND_RE.finditer(cand_m.group(1)):
                    wid, dp, logit, overlap, ptok, dblk = cm.groups()
                    candidates.append(
                        dict(
                            worker_id=wid, dp_rank=int(dp), logit=float(logit),
                            overlap=float(overlap), prefill_tok=int(ptok),
                            decode_blk=int(dblk),
                        )
                    )
                fields_str = fields_str[: cand_m.start()]
            kv = {}
            for km in KV_RE.finditer(fields_str):
                k, v = km.groups()
                kv[k] = v.strip('"')
            records.append(dict(
                ts=ts,
                request_id=kv.get("request_id"),
                worker_type=kv.get("worker_type"),
                isl_tokens=int(kv.get("isl_tokens", 0)),
                chosen_worker_id=kv.get("chosen_worker_id"),
                chosen_logit=float(kv.get("chosen_logit", "nan")),
                candidates=candidates,
            ))
    return records


def compute_components(rec, weights):
    """Returns, per candidate: base, queued_prefill/queued_decode (gross,
    pre-credit), credit_applied (subtracted off the top), gross (=sum of the
    three, pre-credit), and logit (=gross - credit_applied, post-credit,
    matches the logged value)."""
    wt = rec["worker_type"]
    w = weights.get(wt, dict(prefill_load_scale=1.0))
    out = []
    for c in rec["candidates"]:
        if wt == "prefill":
            base = rec["isl_tokens"] / BLOCK_SIZE
            queued_prefill = c["prefill_tok"] / BLOCK_SIZE
            queued_decode = float(c["decode_blk"])
            raw_prefill_combined = base + queued_prefill
            credit = min(c["overlap"], raw_prefill_combined)
            gross = base + queued_prefill + queued_decode
            logit_check = (w["prefill_load_scale"]
                           * max(raw_prefill_combined - c["overlap"], 0)) + queued_decode
        else:  # decode
            queued_decode = float(c["decode_blk"])
            base = c["logit"] - queued_decode  # constant per decision; credit N/A
            queued_prefill = 0.0
            credit = 0.0
            gross = base + queued_prefill + queued_decode
            logit_check = gross
        out.append(dict(
            worker_id=c["worker_id"], logit=c["logit"],
            base=base,
            queued_prefill=queued_prefill,
            queued_decode=queued_decode,
            credit_applied=credit,
            gross=gross,
            logit_check=logit_check,
            chosen=rec["chosen_worker_id"].endswith(c["worker_id"]),
        ))
    return out


COMP_NAMES = ["base", "queued_prefill", "queued_decode"]
COMP_COLORS = {
    "base": "#d55e00",
    "queued_prefill": "#e69f00",
    "queued_decode": "#0072b2",
}
COMP_LABELS = {
    "base": "base (new req's own prefill/decode cost)",
    "queued_prefill": "queued prefill (other reqs on that worker)",
    "queued_decode": "queued decode (other reqs on that worker)",
}


def draw_candidate_bar(ax, x, width, c, chosen_marker="star"):
    """Draws one candidate's bar: solid gross stack [base, queued_prefill,
    queued_decode], a hatched 'credited away' cap for any overlap credit,
    a bold line at the actual logit height, and a chosen-worker marker."""
    bottom = 0.0
    for name in COMP_NAMES:
        val = c[name]
        if val <= 0:
            continue
        ax.bar(x, val, width=width, bottom=bottom, color=COMP_COLORS[name],
               edgecolor="none", zorder=2)
        bottom += val
    if c["credit_applied"] > 1e-9:
        ax.bar(x, c["credit_applied"], width=width, bottom=c["logit"],
               facecolor="#009e73", edgecolor="#00543a", alpha=0.35,
               hatch="///", linewidth=0.5, zorder=3)
    ax.hlines(c["logit"], x - width / 2, x + width / 2,
               color="black", linewidth=1.6, zorder=4)
    if c["chosen"]:
        ax.bar(x, c["logit"], width=width, bottom=0, fill=False,
               edgecolor="red", linewidth=2.2, zorder=5)
        if chosen_marker == "star":
            ax.plot(x, c["gross"] * 1.06 if c["gross"] > 0 else c["logit"] + 1,
                    marker="*", color="red", markersize=11, zorder=6,
                    markeredgecolor="black", markeredgewidth=0.4)
        else:
            ax.text(x, c["gross"] * 1.03 if c["gross"] > 0 else c["logit"] + 1,
                    "chosen", ha="center", fontsize=8, fontweight="bold", zorder=6)


def legend_handles():
    handles = [plt.Rectangle((0, 0), 1, 1, color=COMP_COLORS[n]) for n in COMP_NAMES]
    labels_ = [COMP_LABELS[n] for n in COMP_NAMES]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="#009e73", edgecolor="#00543a",
                                  alpha=0.35, hatch="///"))
    labels_.append("KV cache-hit credit (blocks reused, not charged)")
    handles.append(plt.Line2D([0], [0], color="black", linewidth=1.6))
    labels_.append("actual logit (bold line = true routing cost)")
    return handles, labels_


def plot_grid(records, weights, worker_type, out_path, n_samples=9):
    recs = [r for r in records if r["worker_type"] == worker_type and r["candidates"]]
    if not recs:
        print(f"no {worker_type} records with candidates")
        return
    recs.sort(key=lambda r: r["ts"])
    idxs = sorted(set(int(round(i)) for i in
                       [k * (len(recs) - 1) / (n_samples - 1) for k in range(n_samples)]))
    sample = [recs[i] for i in idxs]

    ncols = 3
    nrows = -(-len(sample) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax, rec in zip(axes.flat, sample):
        comps = compute_components(rec, weights)
        comps.sort(key=lambda c: c["worker_id"])
        labels = [c["worker_id"][-4:] for c in comps]
        for i, c in enumerate(comps):
            draw_candidate_bar(ax, i, 0.7, c, chosen_marker="text")
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(labels)
        ax.set_title(f"t={rec['t_rel']:.1f}s  isl={rec['isl_tokens']}", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel("logit (KV blocks)", fontsize=8)

    for ax in axes.flat[len(sample):]:
        ax.axis("off")

    handles, labels_ = legend_handles()
    fig.legend(handles, labels_, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Stacked logit components per candidate worker — {worker_type} routing\n"
                 f"shortest bold line = chosen worker; hatched cap = KV cache-hit credit",
                 y=1.1, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({len(sample)} sampled decisions of {len(recs)} total)")


def plot_single(records, weights, request_id, worker_type, out_path):
    recs = [r for r in records if r["worker_type"] == worker_type
            and r["request_id"] == request_id]
    if not recs:
        print(f"no record for request_id={request_id} worker_type={worker_type}")
        return
    rec = recs[0]
    comps = compute_components(rec, weights)
    comps.sort(key=lambda c: c["worker_id"])
    labels = [c["worker_id"][-4:] for c in comps]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, c in enumerate(comps):
        draw_candidate_bar(ax, i, 0.6, c, chosen_marker="text")
        ax.text(i, c["logit"] / 2 if c["logit"] > 0 else 0.5,
                f"{c['logit']:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("logit (KV blocks)")
    ax.set_title(f"{worker_type} routing — request_id={request_id[:8]}  isl={rec['isl_tokens']}")
    handles, labels_ = legend_handles()
    ax.legend(handles, labels_, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_timeline(records, weights, worker_type, out_path, per_row=20):
    """One strip per `per_row` consecutive decisions: clustered stacked bars,
    one cluster per decision, one bar per candidate worker within a cluster.
    The chosen worker's bar gets a thick red outline + a red star above it,
    so it's identifiable even when tied in height with other candidates."""
    recs = [r for r in records if r["worker_type"] == worker_type and r["candidates"]]
    if not recs:
        print(f"no {worker_type} records with candidates")
        return
    recs.sort(key=lambda r: r["ts"])

    worker_ids = sorted({c["worker_id"] for r in recs for c in r["candidates"]})
    n_workers = len(worker_ids)
    w_idx = {w: i for i, w in enumerate(worker_ids)}

    n_rows = -(-len(recs) // per_row)
    fig, axes = plt.subplots(n_rows, 1, figsize=(max(14, per_row * 0.9), 3.2 * n_rows),
                              squeeze=False)
    axes = axes[:, 0]

    cluster_w = 0.82
    bar_w = cluster_w / n_workers

    for row, ax in enumerate(axes):
        chunk = recs[row * per_row:(row + 1) * per_row]
        for i, rec in enumerate(chunk):
            comps = compute_components(rec, weights)
            comps_by_worker = {c["worker_id"]: c for c in comps}
            for w in worker_ids:
                c = comps_by_worker.get(w)
                if c is None:
                    continue
                x = i + (w_idx[w] - (n_workers - 1) / 2) * bar_w
                draw_candidate_bar(ax, x, bar_w * 0.95, c, chosen_marker="star")

        ax.set_xlim(-0.5, len(chunk) - 0.5)
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([f"{r['t_rel']:.1f}s" for r in chunk],
                            rotation=90, fontsize=7)
        ax.set_ylabel("logit\n(KV blocks)", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("time (s, relative to first decision)")

    handles, labels_ = legend_handles()
    handles.append(plt.Line2D([0], [0], marker="*", color="red", linestyle="",
                               markersize=11, markeredgecolor="black"))
    labels_.append("chosen worker (red star + red outline)")
    fig.legend(handles, labels_, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.0 + 0.9 / (3.2 * n_rows)))
    fig.suptitle(
        f"Stacked logit components per candidate worker over time — {worker_type} routing\n"
        f"each cluster = one routing decision; {n_workers} bars = candidates "
        f"({', '.join(w[-4:] for w in worker_ids)}); bold line = actual logit; "
        f"hatched cap = KV cache-hit credit; red star = chosen worker",
        y=1.0 + 1.8 / (3.2 * n_rows), fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path} ({len(recs)} decisions, {n_rows} rows of {per_row})")


def main():
    log_dir = f"{RUN_DIR}/logs"
    out_dir = f"{RUN_DIR}/analysis"
    os.makedirs(out_dir, exist_ok=True)
    log_paths = sorted(glob.glob(f"{log_dir}/frontend-*.log"))

    weights = {}
    all_records = []
    for p in log_paths:
        weights.update(parse_weights(p))
        recs = parse_decisions(p)
        all_records.extend(recs)

    all_records.sort(key=lambda r: r["ts"])
    t0 = all_records[0]["ts"]
    for r in all_records:
        r["t_rel"] = (r["ts"] - t0).total_seconds()

    if len(sys.argv) >= 3 and sys.argv[1] == "--request":
        request_id = sys.argv[2]
        worker_type = sys.argv[3] if len(sys.argv) > 3 else "prefill"
        plot_single(all_records, weights, request_id, worker_type,
                    f"{out_dir}/5_stacked_logit_{worker_type}_{request_id[:8]}.png")
        return

    plot_grid(all_records, weights, "prefill", f"{out_dir}/5_stacked_logit_grid_prefill.png")
    plot_grid(all_records, weights, "decode", f"{out_dir}/5_stacked_logit_grid_decode.png")
    plot_timeline(all_records, weights, "prefill", f"{out_dir}/6_stacked_logit_timeline_prefill.png")
    plot_timeline(all_records, weights, "decode", f"{out_dir}/6_stacked_logit_timeline_decode.png")


if __name__ == "__main__":
    main()
