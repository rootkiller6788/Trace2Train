from typing import List
from .schema import TraceEvent


def validate_trace(events: List[TraceEvent]) -> List[str]:
    errors = []
    req_ids = {}
    for evt in events:
        rid = evt.req_id
        if rid not in req_ids:
            req_ids[rid] = []
        req_ids[rid].append(evt.event)

    for rid, event_types in req_ids.items():
        if "reject" in event_types and "complete" in event_types:
            errors.append(f"req {rid}: both rejected and completed")
        if "enqueue" not in event_types:
            errors.append(f"req {rid}: missing enqueue")
        if "reject" not in event_types and "complete" not in event_types:
            errors.append(f"req {rid}: no terminal event (reject/complete)")

    return errors
