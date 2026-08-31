#!/usr/bin/env python3
"""Validate bounded lifecycle facts in an OTLP JSONL trace export.

This is a development-only analyzer.  It intentionally reports UNKNOWN when a
stage is conditional or unsupported rather than treating an absent span as a
passing result.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

LIFECYCLE_NAMES = {
    "request.lifecycle",
    "request.preprocessing",
    "router.queue",
    "router.selection",
    "worker.admission",
    "request.dispatch",
    "worker.operation.prefill",
    "worker.operation.decode",
    "response.streaming",
    "response.streaming.prefill",
    "response.streaming.decode",
}


def value(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in raw:
            return raw[key]
    return raw


def attrs(span: dict[str, Any]) -> dict[str, Any]:
    return {
        item["key"]: value(item.get("value")) for item in span.get("attributes", [])
    }


def iter_spans(value_: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value_, dict):
        if {"name", "spanId", "startTimeUnixNano", "endTimeUnixNano"} <= value_.keys():
            yield value_
        for child in value_.values():
            yield from iter_spans(child)
    elif isinstance(value_, list):
        for child in value_:
            yield from iter_spans(child)


def duration_ms(span: dict[str, Any]) -> float:
    return (int(span["endTimeUnixNano"]) - int(span["startTimeUnixNano"])) / 1_000_000


def number(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def same_destination(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return str(left.get("dynamo.dispatch.destination.worker.id")) == str(
        right.get("dynamo.router.selected.worker.id")
    ) and str(left.get("dynamo.dispatch.destination.dp.rank")) == str(
        right.get("dynamo.router.selected.dp.rank")
    )


@dataclass
class Result:
    passed: int = 0
    violated: int = 0
    unknown: int = 0

    def add(self, state: str) -> None:
        setattr(self, state, getattr(self, state) + 1)

    def text(self) -> str:
        if self.violated:
            state = "VIOLATED"
        elif self.passed:
            state = "HOLDS"
        else:
            state = "UNKNOWN"
        return f"{state:9} pass={self.passed} violated={self.violated} unknown={self.unknown}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--expect-queue-outcome",
        action="append",
        choices=("admitted", "cancelled", "rejected"),
        help="fail validation unless this queue outcome is present at least once",
    )
    args = parser.parse_args()

    spans: list[dict[str, Any]] = []
    batches = 0
    for line in args.trace.read_text().splitlines():
        if not line.strip():
            continue
        batches += 1
        spans.extend(iter_spans(json.loads(line)))
    for span in spans:
        span["_attrs"] = attrs(span)

    lifecycle = [span for span in spans if span["name"] in LIFECYCLE_NAMES]
    by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in spans:
        by_trace[span.get("traceId", "")].append(span)

    roots = [span for span in lifecycle if span["name"] == "request.lifecycle"]
    request_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace_id, trace_spans in by_trace.items():
        for span in trace_spans:
            request_id = span["_attrs"].get("dynamo.request.id") or span["_attrs"].get(
                "request_id"
            )
            if request_id:
                request_groups[(trace_id, str(request_id))].append(span)

    results = {
        "http_lifecycle_parentage": Result(),
        "http_lifecycle_start_bounds": Result(),
        "frontend_ttft_bounds": Result(),
        "dispatch_selection_path": Result(),
        "router_selection_filter_accounting": Result(),
        "router_selection_investigation_detail": Result(),
        "router_queue_fact_completeness": Result(),
        "router_queue_terminal_causality": Result(),
        "router_queue_expected_outcome": Result(),
    }

    for root in roots:
        root_attrs = root["_attrs"]
        key = (root.get("traceId", ""), str(root_attrs["dynamo.request.id"]))
        group = request_groups[key]
        http = [span for span in group if span["name"] == "http-request"]
        if len(http) != 1:
            results["http_lifecycle_parentage"].add("violated")
            results["http_lifecycle_start_bounds"].add("unknown")
            results["frontend_ttft_bounds"].add("unknown")
            continue
        frontend = http[0]
        results["http_lifecycle_parentage"].add(
            "passed"
            if root.get("parentSpanId") == frontend.get("spanId")
            else "violated"
        )
        # `http-request` ends once a streaming response is handed to the
        # transport, whereas request.lifecycle owns the stream's terminal
        # state.  Consequently a valid lifecycle span may outlive its HTTP
        # parent.  The invariant is the causal start boundary, not equality
        # or temporal nesting of their full durations.
        start_ns = int(root["startTimeUnixNano"])
        http_start_ns = int(frontend["startTimeUnixNano"])
        http_end_ns = int(frontend["endTimeUnixNano"])
        results["http_lifecycle_start_bounds"].add(
            "passed" if http_start_ns <= start_ns <= http_end_ns else "violated"
        )
        output_tokens = number(frontend["_attrs"].get("output_tokens"))
        ttft_ms = number(frontend["_attrs"].get("ttft_ms"))
        if output_tokens is None or output_tokens <= 0:
            results["frontend_ttft_bounds"].add("unknown")
        elif ttft_ms is None:
            results["frontend_ttft_bounds"].add("violated")
        else:
            results["frontend_ttft_bounds"].add(
                "passed" if 0.0 <= ttft_ms <= duration_ms(root) else "violated"
            )

    for (_, _), group in request_groups.items():
        selections = [span for span in group if span["name"] == "router.selection"]
        dispatches = [span for span in group if span["name"] == "request.dispatch"]
        if not dispatches:
            continue
        for dispatch in dispatches:
            facts = dispatch["_attrs"]
            required = [
                "dynamo.lifecycle.detail_schema",
                "dynamo.component",
                "dynamo.dispatch.destination.worker.id",
                "dynamo.dispatch.destination.dp.rank",
                "dynamo.dispatch.route",
                "dynamo.dispatch.result",
            ]
            if any(key not in facts for key in required):
                results["dispatch_selection_path"].add("violated")
            elif facts["dynamo.component"] != "frontend":
                results["dispatch_selection_path"].add("violated")
            elif facts["dynamo.dispatch.result"] != "accepted":
                results["dispatch_selection_path"].add("unknown")
            elif any(
                same_destination(facts, selection["_attrs"]) for selection in selections
            ):
                results["dispatch_selection_path"].add("passed")
            else:
                results["dispatch_selection_path"].add("violated")

    for selection in [span for span in lifecycle if span["name"] == "router.selection"]:
        facts = selection["_attrs"]
        required = [
            "dynamo.lifecycle.detail_schema",
            "dynamo.router.candidate.count",
            "dynamo.router.candidate.eligible.count",
            "dynamo.router.candidate.filtered.count",
            "dynamo.router.candidate.filtered.not_allowed",
            "dynamo.router.candidate.filtered.constraints",
            "dynamo.router.candidate.filtered.overloaded",
            "dynamo.router.candidate.filtered.unavailable",
        ]
        if (
            any(key not in facts for key in required)
            or facts.get("dynamo.lifecycle.detail_schema") != "router_selection.v1"
        ):
            results["router_selection_filter_accounting"].add("violated")
        else:
            candidate = number(facts["dynamo.router.candidate.count"])
            eligible = number(facts["dynamo.router.candidate.eligible.count"])
            filtered = number(facts["dynamo.router.candidate.filtered.count"])
            parts = sum(number(facts[key]) or 0 for key in required[4:])
            results["router_selection_filter_accounting"].add(
                "passed" if candidate == eligible and filtered == parts else "violated"
            )

        detail = facts.get("dynamo.router.candidates.top_k")
        if detail is None:
            results["router_selection_investigation_detail"].add("unknown")
            continue
        try:
            candidates = json.loads(str(detail))
            selected = [
                candidate for candidate in candidates if candidate.get("selected")
            ]
            scores = [float(candidate["score"]) for candidate in candidates]
            selected_facts = (
                str(facts["dynamo.router.selected.worker.id"]),
                str(facts["dynamo.router.selected.dp.rank"]),
            )
            selected_detail = [
                candidate
                for candidate in selected
                if (str(candidate["worker_id"]), str(candidate["dp_rank"]))
                == selected_facts
            ]
            multiplier_ok = all(
                math.isclose(
                    float(candidate["score"]),
                    float(candidate["base_score"])
                    * float(candidate["preferred_taint_multiplier"]),
                    abs_tol=1e-5,
                )
                for candidate in candidates
            )
            valid = (
                len(candidates) <= 4
                and len(selected) == 1
                and len(selected_detail) == 1
                and scores == sorted(scores)
                and multiplier_ok
            )
            results["router_selection_investigation_detail"].add(
                "passed" if valid else "violated"
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            results["router_selection_investigation_detail"].add("violated")

    queues = [span for span in lifecycle if span["name"] == "router.queue"]
    root_outcomes = {
        (root.get("traceId", ""), str(root["_attrs"]["dynamo.request.id"])): root[
            "_attrs"
        ].get("dynamo.request.terminal.outcome")
        for root in roots
    }
    if not queues:
        results["router_queue_fact_completeness"].add("unknown")
        results["router_queue_terminal_causality"].add("unknown")
    for queue in queues:
        facts = queue["_attrs"]
        required = [
            "dynamo.lifecycle.detail_schema",
            "dynamo.router.queue.class",
            "dynamo.router.queue.policy",
            "dynamo.router.queue.depth.in",
            "dynamo.router.queue.deferred",
            "dynamo.router.queue.outcome",
        ]
        if any(key not in facts for key in required):
            results["router_queue_fact_completeness"].add("violated")
        elif (
            facts["dynamo.router.queue.outcome"]
            in {"admitted", "cancelled", "rejected"}
            and "dynamo.router.queue.depth.out" not in facts
        ):
            results["router_queue_fact_completeness"].add("violated")
        elif (
            facts["dynamo.router.queue.outcome"] in {"cancelled", "rejected"}
            and "dynamo.router.queue.reason" not in facts
        ):
            results["router_queue_fact_completeness"].add("violated")
        else:
            results["router_queue_fact_completeness"].add("passed")

        outcome = facts.get("dynamo.router.queue.outcome")
        if outcome not in {"cancelled", "rejected"}:
            continue
        key = (queue.get("traceId", ""), str(facts.get("dynamo.request.id", "")))
        # Queue cancellation/rejection is a local event.  Its end-to-end
        # meaning is validated only when the root terminal outcome for the
        # same trace/request agrees with it.
        results["router_queue_terminal_causality"].add(
            "passed" if root_outcomes.get(key) == outcome else "violated"
        )

    terminal_queues = [
        queue
        for queue in queues
        if queue["_attrs"].get("dynamo.router.queue.outcome")
        in {"cancelled", "rejected"}
    ]
    if not terminal_queues and queues:
        results["router_queue_terminal_causality"].add("unknown")
    observed_queue_outcomes = {
        str(queue["_attrs"].get("dynamo.router.queue.outcome")) for queue in queues
    }
    for expected in args.expect_queue_outcome or []:
        results["router_queue_expected_outcome"].add(
            "passed" if expected in observed_queue_outcomes else "violated"
        )
    if not args.expect_queue_outcome:
        results["router_queue_expected_outcome"].add("unknown")

    print(
        f"batches={batches} spans={len(spans)} lifecycle_spans={len(lifecycle)} roots={len(roots)}"
    )
    print(
        "span_counts="
        + json.dumps(Counter(span["name"] for span in lifecycle), sort_keys=True)
    )
    print("invariants:")
    for name, result in results.items():
        print(f"  {name}: {result.text()}")
    return 1 if any(result.violated for result in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
