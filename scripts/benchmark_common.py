"""Shared HTTP and statistical helpers for the benchmark runners."""

from __future__ import annotations

import json
import math
import os
import urllib.request
from typing import Any


def request(
    base: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
):
    """Open an API request with the repository's optional benchmark auth."""
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("BENCH_API_KEY")
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    return urllib.request.urlopen(
        urllib.request.Request(base.rstrip("/") + path, data=data, headers=headers),
        timeout=timeout,
    )


def json_request(
    base: str,
    path: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3600,
) -> dict[str, Any]:
    """Read one JSON response through the same request path as streaming calls."""
    with request(base, path, body, timeout) as response:
        return json.load(response)


def percentile(values: list[float], fraction: float) -> float | None:
    """Return the nearest-rank percentile used by every benchmark runner."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
