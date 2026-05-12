import json
from pathlib import Path
from typing import List
from ..trace.schema import RequestTrace


def build_sft_samples(
    traces: List[RequestTrace],
    output_path: str,
) -> None:
    samples = []
    for t in traces:
        if t.status != "success":
            continue
        instruction = (
            f"Process request id={t.req_id} with "
            f"{t.requested_tokens} tokens using "
            f"{t.kv_bytes} bytes of KV cache."
        )
        completion = (
            f"Schedule request {t.req_id}: allocate {t.kv_bytes} bytes, "
            f"decode {t.total_tokens} tokens, "
            f"latency {t.total_latency_us / 1000:.1f} ms."
        )
        samples.append({
            "instruction": instruction,
            "output": completion,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"SFT samples: {len(samples)} written to {output_path}")
