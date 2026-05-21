import requests
from config.settings import BRAVE_SEARCH_API_KEY


def search_web(query: str, num_results: int = 5) -> list[dict]:
    """Search the web. Uses Brave Search if API key is set, otherwise DuckDuckGo."""
    num_results = max(1, min(10, num_results))

    if BRAVE_SEARCH_API_KEY:
        return _brave_search(query, num_results)
    return _ddg_search(query, num_results)


def _brave_search(query: str, num_results: int) -> list[dict]:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params={"q": query, "count": num_results},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("web", {}).get("results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "snippet": item.get("description", ""),
            "url": item.get("url", ""),
        })
    return results


def _ddg_search(query: str, num_results: int) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=num_results))
        return [
            {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
            for r in raw
        ]
    except ImportError:
        return _ddg_instant(query, num_results)


def _ddg_instant(query: str, num_results: int) -> list[dict]:
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        timeout=10,
    )
    data = resp.json()
    results = []
    if data.get("Abstract"):
        results.append({
            "title": data.get("AbstractSource", "DuckDuckGo"),
            "snippet": data["Abstract"],
            "url": data.get("AbstractURL", ""),
        })
    for topic in data.get("RelatedTopics", []):
        if isinstance(topic, dict) and topic.get("Text"):
            results.append({
                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                "snippet": topic["Text"],
                "url": topic.get("FirstURL", ""),
            })
        if len(results) >= num_results:
            break
    return results[:num_results]
