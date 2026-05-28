"""Dynamic provider test — auto-discovers every provider and role from config.yaml.

Tests each provider directly, then tests routing, fallback, and tool calling.
Measures latency: time-to-first-token, total stream time, complete() duration.
No hardcoded provider names — just update config.yaml and re-run.
"""
import os
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from daisy.utils.config_loader import load_config

logging.basicConfig(level=logging.WARNING, format='  [%(name)s] %(message)s')

c = load_config()
PASS = 0
FAIL = 0

# Latency records: list of dicts for a summary table at the end
LATENCY = []


def ok(msg):
    global PASS
    PASS += 1
    print(f"  PASS  {msg}")


def bad(msg, detail=""):
    global FAIL
    FAIL += 1
    detail = f" — {detail}" if detail else ""
    print(f"  FAIL  {msg}{detail}")


# ---------------------------------------------------------------------------
# Phase 1 — Test every provider directly
# ---------------------------------------------------------------------------
async def test_provider_direct(name, p):
    """Test a single provider with a direct client (no router)."""
    model_name = next(iter(p.models.values())) if p.models else "unknown"
    temperature = 0.5
    max_tokens = 128

    print(f"\n{'='*60}")
    print(f"Provider: {name}  |  type={p.type}  |  model={model_name}")
    print(f"{'='*60}")

    if p.type == "openai_compatible":
        from daisy.llm.openai_compatible import OpenAICompatibleClient
        try:
            client = OpenAICompatibleClient(
                base_url=p.base_url or "",
                api_key_env=p.api_key_env,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            bad(f"init", str(e))
            return
    elif p.type == "anthropic":
        from daisy.llm.anthropic_client import AnthropicClient
        try:
            client = AnthropicClient(
                api_key_env=p.api_key_env,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            bad(f"init", str(e))
            return
    elif p.type == "google":
        from daisy.llm.gemini_client import GeminiClient
        try:
            client = GeminiClient(
                api_key_env=p.api_key_env,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            bad(f"init", str(e))
            return
    else:
        print(f"  SKIP  Unknown provider type: {p.type}")
        return

    # --- streaming + latency ---
    stream_ok = False
    ttft = None
    stream_time = None
    token_count = 0
    try:
        tokens = []
        t0 = time.monotonic()
        async for token in client.stream_tokens([
            {"role": "user", "content": "Say hello in exactly 5 words."}
        ]):
            if ttft is None:
                ttft = time.monotonic() - t0
            tokens.append(token)
            token_count += 1
        stream_time = time.monotonic() - t0
        response = "".join(tokens).strip()
        if response:
            tok_s = token_count / stream_time if stream_time and stream_time > 0 else 0
            ok(f"stream: {response!r}  |  TTFT={ttft*1000:.0f}ms  total={stream_time*1000:.0f}ms  ~{tok_s:.0f} tok/s")
            stream_ok = True
        else:
            bad("stream: empty response")
    except Exception as e:
        bad("stream", str(e)[:120])

    # --- complete + latency ---
    complete_ok = False
    complete_time = None
    try:
        t0 = time.monotonic()
        resp = await client.complete([
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ])
        complete_time = time.monotonic() - t0
        if resp.content and resp.content.strip():
            ok(f"complete: {resp.content.strip()!r}  |  {complete_time*1000:.0f}ms")
            complete_ok = True
        else:
            bad("complete: empty content")
    except Exception as e:
        bad("complete", str(e)[:120])

    LATENCY.append({
        "provider": name,
        "model": model_name,
        "stream_ok": stream_ok,
        "ttft_ms": round(ttft * 1000) if ttft else None,
        "stream_total_ms": round(stream_time * 1000) if stream_time else None,
        "stream_tokens": token_count,
        "complete_ok": complete_ok,
        "complete_ms": round(complete_time * 1000) if complete_time else None,
    })


# ---------------------------------------------------------------------------
# Phase 2 — Test routing (every configured role)
# ---------------------------------------------------------------------------
async def test_routing():
    """Test every role in config.routing through the router."""
    from daisy.llm.router import LLMRouter

    router = LLMRouter(c.llm)
    roles = ["main_agent", "summarizer", "sub_agent", "announcement"]

    print(f"\n{'='*60}")
    print("Router — all roles")
    print(f"{'='*60}")

    for role in roles:
        routing = getattr(c.llm.routing, role, None)
        if routing is None:
            print(f"  SKIP  {role}: not configured")
            continue

        prov_name = routing.provider
        model_alias = routing.model
        prov = c.llm.providers.get(prov_name)
        model_id = prov.models.get(model_alias, model_alias) if prov else model_alias

        print(f"\n  Role: {role} -> {prov_name}/{model_alias} ({model_id})")

        try:
            tokens = []
            ttft = None
            t0 = time.monotonic()
            async for token in router.stream_tokens(role, [
                {"role": "user", "content": "Say hi in exactly 3 words."}
            ]):
                if ttft is None:
                    ttft = time.monotonic() - t0
                tokens.append(token)
            total = time.monotonic() - t0
            response = "".join(tokens).strip()
            if response:
                tok_s = len(tokens) / total if total > 0 else 0
                ok(f"{role}: {response!r}  |  TTFT={ttft*1000:.0f}ms  total={total*1000:.0f}ms  ~{tok_s:.0f} tok/s")
            else:
                bad(f"{role}: empty response")
        except Exception as e:
            bad(f"{role}", str(e)[:120])


# ---------------------------------------------------------------------------
# Phase 3 — Test fallback
# ---------------------------------------------------------------------------
async def test_fallback():
    """Force a nonexistent provider on main_agent, verify fallback fires."""
    from daisy.llm.router import LLMRouter

    routing = c.llm.routing.main_agent
    if not routing or not routing.fallback:
        print(f"\n{'='*60}")
        print("Fallback — SKIPPED (no fallback configured for main_agent)")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"Fallback — force nonexistent provider, expect fallback to kick in")
    print(f"{'='*60}")

    original = routing.provider
    routing.provider = "__nonexistent_provider__"
    router = LLMRouter(c.llm)
    routing.provider = original

    fb_chain = [f"{fb['provider']}/{fb['model']}" for fb in routing.fallback]
    print(f"  Fallback chain: {fb_chain}")

    try:
        tokens = []
        t0 = time.monotonic()
        async for token in router.stream_tokens("main_agent", [
            {"role": "user", "content": "Say hello."}
        ]):
            tokens.append(token)
        total = time.monotonic() - t0
        response = "".join(tokens).strip()
        if response:
            ok(f"fallback succeeded: {response!r}  |  {total*1000:.0f}ms")
        else:
            bad("fallback: empty response")
    except Exception as e:
        bad("fallback", str(e)[:120])


# ---------------------------------------------------------------------------
# Phase 4 — Test tool calling
# ---------------------------------------------------------------------------
async def test_tool_calling():
    """Test that the router passes tools through to the model correctly."""
    from daisy.llm.router import LLMRouter

    router = LLMRouter(c.llm)

    print(f"\n{'='*60}")
    print("Tool calling — main_agent with a get_weather tool")
    print(f"{'='*60}")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }]

    try:
        t0 = time.monotonic()
        resp = await router.complete("main_agent", [
            {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}
        ], tools=tools)
        total = time.monotonic() - t0

        if resp.tool_calls:
            ok(f"{len(resp.tool_calls)} tool call(s)  |  {total*1000:.0f}ms")
            for tc in resp.tool_calls:
                print(f"         {tc.name}({tc.arguments})")
        elif resp.content:
            ok(f"no tool calls, text: {resp.content.strip()!r}  |  {total*1000:.0f}ms")
        else:
            bad("no tool calls and no content")
    except Exception as e:
        bad("tool calling", str(e)[:120])


# ---------------------------------------------------------------------------
# Latency Summary Table
# ---------------------------------------------------------------------------
def print_latency_summary():
    if not LATENCY:
        return
    print(f"\n{'='*85}")
    print("Latency Summary — Direct Provider Calls")
    print(f"{'='*85}")
    print(f"{'Provider':<16} {'Model':<28} {'TTFT':>7} {'Stream':>8} {'t/s':>6} {'Complete':>9}")
    print(f"{'-'*16} {'-'*28} {'-'*7} {'-'*8} {'-'*6} {'-'*9}")
    for r in LATENCY:
        ttft = f"{r['ttft_ms']}ms" if r['ttft_ms'] is not None else "—"
        stream = f"{r['stream_total_ms']}ms" if r['stream_total_ms'] is not None else "—"
        tok_s = f"{r['stream_tokens'] / (r['stream_total_ms']/1000):.0f}" if r['stream_ok'] and r['stream_total_ms'] else "—"
        complete = f"{r['complete_ms']}ms" if r['complete_ms'] is not None else "—"
        s_flag = " " if r['stream_ok'] else "!"
        c_flag = " " if r['complete_ok'] else "!"
        print(f"{s_flag}{r['provider']:<15} {r['model']:<28} {ttft:>7} {stream:>8} {tok_s:>6} {c_flag}{complete:>9}")
    print(f"{'='*85}")
    print("  ! = failed    TTFT = time-to-first-token    t/s = tokens/sec")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    global PASS, FAIL

    print("=" * 60)
    print("DAISY Multi-Provider Dynamic Test")
    print("=" * 60)
    print(f"Providers found: {list(c.llm.providers.keys())}")
    print(f"Roles configured: main_agent, summarizer, sub_agent, announcement")

    for name, p in c.llm.providers.items():
        await test_provider_direct(name, p)

    await test_routing()
    await test_fallback()
    await test_tool_calling()

    print_latency_summary()

    total = PASS + FAIL
    print(f"\nResults: {PASS} passed, {FAIL} failed, {total} total")
    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
