# rag/websearch.py
# Minimal web search adapter. Supports SERPER (Google) or Bing v7.
# If not configured, returns [] so the pipeline runs vector-only.

import os, requests
from typing import List, Dict, Any

# Choose provider via env:
#   WEB_PROVIDER=serper + SERPER_API_KEY=xxxx
#   WEB_PROVIDER=bing   + BING_API_KEY=xxxx
PROVIDER = os.getenv("WEB_PROVIDER", "").strip().lower()
TIMEOUT  = float(os.getenv("WEB_TIMEOUT_S", "10"))

def _serper_search(q: str, k: int) -> List[Dict[str, Any]]:
    api = os.getenv("SERPER_API_KEY", "").strip()
    if not api:
        return []
    try:
        r = requests.post(
            "https://google.serper.dev/search",
            json={"q": q, "num": min(k, 10)},
            headers={"X-API-KEY": api, "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
        items = []
        for it in (data.get("organic", []) or [])[:k]:
            # Serper sometimes includes a 'date' key; otherwise we omit
            dt = ""
            if isinstance(it.get("date"), str):
                dt = it.get("date", "")[:10]
            items.append({
                "title": it.get("title","").strip(),
                "url": it.get("link","").strip(),
                "snippet": it.get("snippet","").strip(),
                "date": dt,
            })
        return items
    except Exception:
        return []

def _bing_search(q: str, k: int) -> List[Dict[str, Any]]:
    api = os.getenv("BING_API_KEY", "").strip()
    if not api:
        return []
    try:
        r = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": q, "count": k, "responseFilter": "Webpages"},
            headers={"Ocp-Apim-Subscription-Key": api},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json() or {}
        items = []
        for it in (data.get("webPages", {}).get("value", []) or [])[:k]:
            items.append({
                "title": it.get("name","").strip(),
                "url": it.get("url","").strip(),
                "snippet": it.get("snippet","").strip(),
                "date": it.get("dateLastCrawled","")[:10] if it.get("dateLastCrawled") else "",
            })
        return items
    except Exception:
        return []

def web_search(q: str, k: int = 6) -> List[Dict[str, Any]]:
    if PROVIDER == "serper":
        return _serper_search(q, k)
    if PROVIDER == "bing":
        return _bing_search(q, k)
    return []  # not configured
