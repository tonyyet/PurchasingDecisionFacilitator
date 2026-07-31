"""Firecrawl web search — bare HTTP (no pip package), cached, fail-open.

Key: env FIRECRAWL_API_KEY (never printed/logged).
Latency ~15s per query -> 30s timeout, in-memory cache TTL 10min, failure returns None.
"""
import os
import time

import requests

_SEARCH_CACHE: dict = {}  # query -> (ts, results)
_TTL = 600


def web_search(query: str, n: int = 5, timeout: float = 30.0):
    """Return list of {title,url,text} or None on failure. Never raises."""
    now = time.time()
    hit = _SEARCH_CACHE.get(query)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    key = os.environ.get("FIRECRAWL_API_KEY")
    if not key:
        return None
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v1/search",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": query, "limit": n},
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        if not data.get("success"):
            return None
        results = []
        for it in (data.get("data") or [])[:n]:
            text = (it.get("description") or it.get("markdown") or "")[:500]
            results.append({"title": it.get("title", ""), "url": it.get("url", ""), "text": text})
        if results:
            _SEARCH_CACHE[query] = (now, results)
        return results or None
    except Exception:
        return None


def search_many(queries, n: int = 5, timeout: float = 30.0):
    """Sequential to avoid rate limiting; returns dict query -> results-or-None."""
    out = {}
    for q in queries:
        out[q] = web_search(q, n=n, timeout=timeout)
    return out
