import json
from pathlib import Path
from typing import List
from ..trace.schema import RequestTrace


def build_dpo_pairs(
    traces: List[RequestTrace],
    output_path: str,
) -> None:
    successes = [t for t in traces if t.status == "success"]
    failures = [t for t in traces if t.status == "failure"]
    pairs = []

    for i, s in enumerate(successes):
        if i < len(failures):
            f = failures[i]
        else:
            f = failures[0] if failures else s

        prompt = (
            f"Schedule request with params: "
            f"max_tokens={s.requested_tokens}"
        )
        chosen = (
            f"Accept and allocate {s.kv_bytes} bytes. "
            f"Result: latency={s.total_latency_us / 1000:.1f} ms, "
            f"tokens={s.total_tokens}"
        )
        rejected = (
            f"Reject due to {f.reason or 'resource_constraint'}"
            if f.status == "failure"
            else "Accept (suboptimal)"
        )
        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"DPO pairs: {len(pairs)} written to {output_path}")
