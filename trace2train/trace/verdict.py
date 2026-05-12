"""Smart trace verifier: labels requests as accept/reject with reasons."""

from typing import List, Dict
from ..trace.schema import RequestTrace


class VerdictLabeler:
    """Label requests based on scheduling outcomes."""

    @staticmethod
    def label(traces: List[RequestTrace]) -> List[Dict]:
        results = []

        for t in traces:
            verdict = "accept" if t.status == "success" else "reject"

            if verdict == "reject":
                reasons = set()
                for evt in t.events:
                    if hasattr(evt, "reason") and evt.reason:
                        reasons.add(evt.reason)
                verdict_reason = ", ".join(reasons) if reasons else "unknown"
            else:
                verdict_reason = "scheduled_successfully"

            avg_latency_per_token = (
                t.total_latency_us / max(t.total_tokens, 1) / 1000.0
            )
            kv_efficiency = (
                t.kv_bytes / max(t.requested_tokens, 1)
            )

            results.append({
                "req_id": t.req_id,
                "verdict": verdict,
                "verdict_reason": verdict_reason,
                "requested_tokens": t.requested_tokens,
                "actual_tokens": t.total_tokens,
                "latency_ms": t.total_latency_us / 1000.0,
                "avg_latency_per_token_ms": avg_latency_per_token,
                "kv_bytes": t.kv_bytes,
                "kv_efficiency": kv_efficiency,
            })

        return results

    @staticmethod
    def statistics(labels: List[Dict]) -> Dict:
        total = len(labels)
        accepted = sum(1 for l in labels if l["verdict"] == "accept")
        rejected = total - accepted

        accept_latencies = [
            l["latency_ms"] for l in labels if l["verdict"] == "accept"
        ]
        reject_reasons = {}
        for l in labels:
            if l["verdict"] == "reject":
                reason = l["verdict_reason"]
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / max(total, 1),
            "avg_latency_ms": sum(accept_latencies) / max(len(accept_latencies), 1),
            "max_latency_ms": max(accept_latencies) if accept_latencies else 0,
            "rejection_reasons": reject_reasons,
        }
