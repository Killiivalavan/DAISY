import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def create_app(
    state_machine,
    memory_manager,
    config,
    session_manager,
    event_bridge,
) -> FastAPI:
    """Create and configure the FastAPI application.

    All pipeline components are passed by reference — they live in the
    same process and share the asyncio event loop.
    """
    app = FastAPI(title="D.A.I.S.Y. API", version="2.0.0")

    # REST endpoints
    from daisy.api.routes import create_router

    router = create_router(state_machine, memory_manager, config, session_manager)
    app.include_router(router)

    # WebSocket endpoint
    from daisy.api.ws_handler import create_ws_handler

    ws_handler = create_ws_handler(session_manager, event_bridge, state_machine)
    app.websocket("/ws")(ws_handler)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Static files — PWA frontend served at root.
    # Must be registered LAST. All explicit routes (API, WS, health) take
    # priority because Starlette checks Routes before Mounts by type.
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


async def run_api_server(app: FastAPI, host: str, port: int):
    """Start uvicorn in the same asyncio event loop.

    Uses uvicorn.Server directly rather than uvicorn.run() so it
    shares the loop with the voice pipeline.
    """
    import uvicorn

    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )
    server = uvicorn.Server(server_config)
    logger.info(f"API server starting on {host}:{port}")
    await server.serve()
