"""Small, transport-independent quality and speculative-decoding metrics.

The OpenAI-compatible API does not require speculative-decoding counters to be
present in ``usage``.  These helpers therefore treat them as optional evidence:
the benchmark remains usable on stock vLLM, while a backend that exposes draft
and accepted token counts gets an auditable acceptance rate instead of an
inferred speedup claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def text_sha256(text: str) -> str:
    """Return a stable digest for the exact streamed text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def speculative_metrics(*payloads: Mapping[str, Any] | None) -> dict[str, Any]:
    """Extract optional draft/accepted counters from API metadata.

    Different engines use slightly different names.  Only integer counters
    are accepted, and accepted tokens must not exceed drafted tokens.  Invalid
    or absent metadata yields an explicit ``available=False`` record rather
    than a fabricated zero acceptance rate.
    """

    drafted_names = (
        "draft_tokens",
        "num_draft_tokens",
        "drafted_tokens",
        "num_drafted_tokens",
    )
    accepted_names = (
        "accepted_tokens",
        "num_accepted_tokens",
        "accepted_draft_tokens",
        "num_accepted_draft_tokens",
    )
    drafted = accepted = None
    pending = [payload for payload in payloads if isinstance(payload, Mapping)]
    while pending:
        payload = pending.pop()
        if not isinstance(payload, Mapping):
            continue
        for value in payload.values():
            if isinstance(value, Mapping):
                pending.append(value)
        for name in drafted_names:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                drafted = value
                break
        for name in accepted_names:
            value = payload.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                accepted = value
                break
    if (
        drafted is None
        or accepted is None
        or drafted < 0
        or accepted < 0
        or accepted > drafted
    ):
        return {
            "available": False,
            "draft_tokens": None,
            "accepted_tokens": None,
            "acceptance_rate": None,
        }
    return {
        "available": True,
        "draft_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def aggregate_speculative(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only rows carrying valid speculative metadata."""

    evidence = [row for row in rows if row.get("speculative", {}).get("available")]
    drafted = sum(int(row["speculative"]["draft_tokens"]) for row in evidence)
    accepted = sum(int(row["speculative"]["accepted_tokens"]) for row in evidence)
    return {
        "available": bool(evidence),
        "requests_with_evidence": len(evidence),
        "draft_tokens": drafted if evidence else None,
        "accepted_tokens": accepted if evidence else None,
        "acceptance_rate": accepted / drafted if evidence and drafted else None,
    }

