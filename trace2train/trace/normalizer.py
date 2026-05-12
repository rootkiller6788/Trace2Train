from typing import List
from .schema import TraceEvent


def normalize_events(events: List[TraceEvent]) -> List[TraceEvent]:
    """Sort by timestamp, deduplicate, fill gaps."""
    events = sorted(events, key=lambda e: e.ts_us)
    seen = set()
    unique = []
    for evt in events:
        key = (evt.ts_us, evt.event, evt.req_id)
        if key not in seen:
            seen.add(key)
            unique.append(evt)
    return unique


def filter_by_status(
    events: List[TraceEvent],
    keep_rejected: bool = True,
) -> List[TraceEvent]:
    """Filter trace events by request outcome."""
    rejected_ids = set()
    for evt in events:
        if evt.event == "reject":
            rejected_ids.add(evt.req_id)

    if keep_rejected:
        return events
    return [e for e in events if e.req_id not in rejected_ids]
