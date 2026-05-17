import asyncio

import httpx
from ddgs import DDGS
from trafilatura import extract


async def web_search(query: str, max_results: int = 5) -> str:
    def _search():
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    results = await asyncio.to_thread(_search)

    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title") or "No title"
        snippet = (r.get("body") or "")[:200]
        url = r.get("href") or ""
        lines.append(f"**{title}**\n{snippet}\n{url}")

    return "\n\n".join(lines)


async def browse_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5),
    ) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            return f"Error: HTTP {e.response.status_code} when fetching {url}"
        except httpx.TimeoutException:
            return "Error: Request timed out."
        except Exception as e:
            return f"Error: {e}"

    text = extract(response.text, output_format="markdown", with_metadata=True)
    if not text:
        return "Could not extract meaningful content from this URL."

    return text[:3000]
