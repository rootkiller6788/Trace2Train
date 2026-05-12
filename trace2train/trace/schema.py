from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TraceEventType(str, Enum):
    ENQUEUE = "enqueue"
    SCHEDULE = "schedule"
    DECODE = "decode"
    COMPLETE = "complete"
    REJECT = "reject"


@dataclass
class TraceEvent:
    ts_us: int
    event: TraceEventType
    req_id: str
    tokens_requested: Optional[int] = None
    batch_id: Optional[int] = None
    kv_alloc_bytes: Optional[int] = None
    step: Optional[int] = None
    latency_us: Optional[int] = None
    total_tokens: Optional[int] = None
    reason: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TraceEvent":
        return cls(
            ts_us=d["ts_us"],
            event=TraceEventType(d["event"]),
            req_id=d.get("req_id", ""),
            tokens_requested=d.get("tokens_requested"),
            batch_id=d.get("batch_id"),
            kv_alloc_bytes=d.get("kv_alloc_bytes"),
            step=d.get("step"),
            latency_us=d.get("latency_us"),
            total_tokens=d.get("total_tokens"),
            reason=d.get("reason"),
        )


@dataclass
class RequestTrace:
    """A single request's complete lifecycle extracted from trace."""
    req_id: str
    requested_tokens: int
    status: str
    total_tokens: int = 0
    total_latency_us: int = 0
    kv_bytes: int = 0
    events: List[TraceEvent] = field(default_factory=list)
