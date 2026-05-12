import json
from pathlib import Path
from typing import List, Dict, Any
from .schema import TraceEvent, RequestTrace


def load_trace(path: str) -> List[TraceEvent]:
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            events.append(TraceEvent.from_dict(d))
    return events


def load_traces(paths: List[str]) -> List[TraceEvent]:
    events = []
    for p in paths:
        events.extend(load_trace(p))
    return events


def group_by_request(events: List[TraceEvent]) -> List[RequestTrace]:
    groups: Dict[str, List[TraceEvent]] = {}
    for evt in events:
        req_id = evt.req_id
        if req_id not in groups:
            groups[req_id] = []
        groups[req_id].append(evt)

    results = []
    for req_id, evts in groups.items():
        status = "unknown"
        requested = 0
        total = 0
        latency = 0
        kv = 0

        for evt in evts:
            if evt.event == "enqueue":
                requested = evt.tokens_requested or 0
            elif evt.event == "complete":
                status = "success"
                total = evt.total_tokens or 0
            elif evt.event == "reject":
                status = "failure"
            elif evt.event == "decode":
                latency += evt.latency_us or 0
            elif evt.event == "schedule":
                kv = evt.kv_alloc_bytes or 0

        results.append(RequestTrace(
            req_id=req_id,
            requested_tokens=requested,
            status=status,
            total_tokens=total,
            total_latency_us=latency,
            kv_bytes=kv,
            events=evts,
        ))

    return results
