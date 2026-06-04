import asyncio
import ipaddress
from urllib.parse import urlparse

import httpx
from ddgs import DDGS
from trafilatura import extract

# IP ranges that browse_url must not fetch
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),       # loopback
    ipaddress.ip_network("10.0.0.0/8"),        # private
    ipaddress.ip_network("172.16.0.0/12"),     # private
    ipaddress.ip_network("192.168.0.0/16"),    # private
    ipaddress.ip_network("169.254.0.0/16"),    # link-local / cloud metadata
    ipaddress.ip_network("::1/128"),           # loopback v6
    ipaddress.ip_network("fc00::/7"),          # unique local v6
    ipaddress.ip_network("fe80::/10"),         # link-local v6
]


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


def _is_ssrf_safe(url: str) -> bool:
    """Return True if the URL's host does not resolve to a blocked IP range."""
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not a raw IP — resolve it (sync; fine for a tool call)
        import socket
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
        except (socket.gaierror, ValueError):
            return False
    return not any(ip in net for net in _SSRF_BLOCKED_NETWORKS)


async def browse_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    if not _is_ssrf_safe(url):
        return "Error: Requests to internal/private hosts are not allowed."

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

    return text[:8000]
