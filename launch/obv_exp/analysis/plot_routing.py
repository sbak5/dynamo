#!/usr/bin/env python3
"""Parse dynamo_kv_router 'Selected worker' log lines and plot logit
components / routing decisions over time."""

import glob
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = os.environ.get("RUN_DIR", "/home/sbak/dynamo_repro/runs/20260708_213527")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
LINE_RE = re.compile(r"^(?P<ts>\S+)\s+INFO.*(?:Selected worker|Routing decision) (?P<fields>.*)$")
KV_RE = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')
CAND_RE = re.compile(
    r"(\d+):(\d+):([\d.]+)\(overlap=([\d.]+),prefill_tok=(\d+),decode_blk=(\d+)\)"
)


def parse_file(path):
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
            # candidates=... is last and contains commas, pull it out first
            cand_m = re.search(r"candidates=(\S+(?:,\S+)*)$", fields_str)
            candidates = []
            if cand_m:
                for cm in CAND_RE.finditer(cand_m.group(1)):
                    wid, dp, logit, overlap, ptok, dblk = cm.groups()
                    candidates.append(
                        dict(
                            worker_id=wid,
                            dp_rank=int(dp),
                            logit=float(logit),
                            overlap=float(overlap),
                            prefill_tok=int(ptok),
                            decode_blk=int(dblk),
                        )
                    )
                fields_str = fields_str[: cand_m.start()]

            kv = {}
            for km in KV_RE.finditer(fields_str):
                k, v = km.groups()
                kv[k] = v.strip('"')

            rec = dict(
                ts=ts,
                request_id=kv.get("request_id"),
                worker_type=kv.get("worker_type"),
                isl_tokens=int(kv.get("isl_tokens", 0)),
                chosen_worker_id=kv.get("chosen_worker_id"),
                chosen_logit=float(kv.get("chosen_logit", "nan")),
                margin=float(kv.get("margin", "nan")),
                raw_prefill_blocks=float(kv.get("raw_prefill_blocks", "nan")),
                overlap_credit_blocks=float(kv.get("overlap_credit_blocks", "nan")),
                decode_cost_blocks=float(kv.get("decode_cost_blocks", "nan")),
                device_overlap_blocks=float(kv.get("device_overlap_blocks", "nan")),
                candidates=candidates,
            )
            records.append(rec)
    return records


def main():
    log_paths = sys.argv[1:] if len(sys.argv) > 1 else glob.glob(
        f"{RUN_DIR}/logs/frontend-*.log"
    )
    all_records = []
    for p in sorted(log_paths):
        recs = parse_file(p)
        for r in recs:
            r["source"] = p.split("/")[-1]
        all_records.extend(recs)

    if not all_records:
        print("No routing decision records found.")
        return

    all_records.sort(key=lambda r: r["ts"])
    t0 = all_records[0]["ts"]
    for r in all_records:
        r["t_rel"] = (r["ts"] - t0).total_seconds()

    prefill = [r for r in all_records if r["worker_type"] == "prefill"]
    decode = [r for r in all_records if r["worker_type"] == "decode"]

    out_dir = f"{RUN_DIR}/analysis"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Plot 1: chosen_logit over time, prefill vs decode ----
    fig, ax = plt.subplots(figsize=(11, 5))
    if prefill:
        ax.scatter(
            [r["t_rel"] for r in prefill],
            [r["chosen_logit"] for r in prefill],
            s=14, alpha=0.6, label="prefill routing", color="#d55e00",
        )
    if decode:
        ax.scatter(
            [r["t_rel"] for r in decode],
            [r["chosen_logit"] for r in decode],
            s=14, alpha=0.6, label="decode routing", color="#0072b2",
        )
    ax.set_xlabel("time (s, relative to first decision)")
    ax.set_ylabel("chosen_logit (cost, lower = preferred)")
    ax.set_title("Chosen routing logit over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/1_chosen_logit_over_time.png", dpi=140)
    plt.close(fig)

    # ---- Plot 2: logit component breakdown (stacked) per decision, prefill ----
    def component_plot(records, worker_type, components, colors):
        if not records:
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        t = [r["t_rel"] for r in records]
        bottom = [0.0] * len(records)
        for comp, color in zip(components, colors):
            vals = [r[comp] for r in records]
            ax.bar(t, vals, bottom=bottom, width=max(t) / max(len(t), 1) * 0.8 if len(t) > 1 else 0.05,
                   label=comp, color=color, alpha=0.85)
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.plot(t, [r["chosen_logit"] for r in records], "k.", ms=3, label="chosen_logit (actual)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("logit component value (KV blocks)")
        ax.set_title(f"Logit component breakdown — {worker_type} routing\n"
                     f"logit = prefill_load_scale*max(raw_prefill_blocks-overlap_credit_blocks,0) + decode_cost_blocks")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/2_logit_components_{worker_type}.png", dpi=140)
        plt.close(fig)

    component_plot(
        prefill, "prefill",
        ["overlap_credit_blocks", "raw_prefill_blocks"],
        ["#009e73", "#d55e00"],
    )
    component_plot(
        decode, "decode",
        ["decode_cost_blocks", "device_overlap_blocks"],
        ["#0072b2", "#009e73"],
    )

    # ---- Plot 3: which worker was chosen over time (per worker_type) ----
    def chosen_worker_plot(records, worker_type):
        if not records:
            return
        worker_ids = sorted({r["chosen_worker_id"] for r in records})
        idx = {w: i for i, w in enumerate(worker_ids)}
        short = {w: w[-4:] for w in worker_ids}  # last 4 digits as label
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.scatter(
            [r["t_rel"] for r in records],
            [idx[r["chosen_worker_id"]] for r in records],
            s=16, alpha=0.7, color="#cc79a7",
        )
        ax.set_yticks(range(len(worker_ids)))
        ax.set_yticklabels([short[w] for w in worker_ids])
        ax.set_xlabel("time (s)")
        ax.set_ylabel("chosen worker (last 4 digits of worker_id)")
        ax.set_title(f"Chosen worker over time — {worker_type} routing")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/3_chosen_worker_{worker_type}.png", dpi=140)
        plt.close(fig)

    chosen_worker_plot(prefill, "prefill")
    chosen_worker_plot(decode, "decode")

    # ---- Plot 4: candidate logit spread vs chosen logit (per worker_type) ----
    def candidate_spread_plot(records, worker_type):
        if not records:
            return
        fig, ax = plt.subplots(figsize=(11, 5))
        t = [r["t_rel"] for r in records]
        mins = [min(c["logit"] for c in r["candidates"]) if r["candidates"] else float("nan") for r in records]
        maxs = [max(c["logit"] for c in r["candidates"]) if r["candidates"] else float("nan") for r in records]
        chosen = [r["chosen_logit"] for r in records]
        ax.fill_between(t, mins, maxs, color="grey", alpha=0.25, label="candidate logit range (min-max)")
        ax.scatter(t, chosen, s=14, color="#d55e00", label="chosen_logit", zorder=3)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("logit")
        ax.set_title(f"Chosen logit vs candidate range — {worker_type} routing\n"
                      f"(chosen sits at/near the min = router picks lowest-cost worker)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/4_candidate_spread_{worker_type}.png", dpi=140)
        plt.close(fig)

    candidate_spread_plot(prefill, "prefill")
    candidate_spread_plot(decode, "decode")

    print(f"Parsed {len(all_records)} routing decisions "
          f"({len(prefill)} prefill, {len(decode)} decode) from {len(log_paths)} file(s).")
    print(f"Plots written to {out_dir}/")


if __name__ == "__main__":
    main()
