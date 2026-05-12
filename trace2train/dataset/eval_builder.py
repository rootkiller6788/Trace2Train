import json
from pathlib import Path
from typing import List
from ..trace.schema import RequestTrace


def build_eval_samples(
    traces: List[RequestTrace],
    output_path: str,
) -> None:
    samples = []
    for t in traces:
        label = "accept" if t.status == "success" else "reject"
        features = {
            "requested_tokens": t.requested_tokens,
            "kv_bytes": t.kv_bytes,
            "latency_us": t.total_latency_us,
        }
        samples.append({
            "req_id": t.req_id,
            "label": label,
            "features": features,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Eval samples: {len(samples)} written to {output_path}")
