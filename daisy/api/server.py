import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class _AuthMiddleware(BaseHTTPMiddleware):
    """Simple bearer-token middleware for /api/* routes.

    If no token is configured, all requests are allowed (dev mode).
    """

    def __init__(self, app, auth_token: str):
        super().__init__(app)
        self._token = auth_token

    async def dispatch(self, request: Request, call_next):
        if not self._token or not request.url.path.startswith("/api"):
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {self._token}":
            return await call_next(request)

        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid API token"},
        )


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

    # CORS — restrict to the PWA's own origin
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://localhost:{config.api.port}",
            f"https://localhost:{config.api.port}",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    # Auth — guard /api/* routes with a bearer token (no-op if unconfigured)
    auth_token = getattr(config.api, "auth_token", "")
    if auth_token:
        logger.info("API authentication enabled (bearer token required)")
    else:
        logger.warning("No api.auth_token configured — API is open (dev mode)")
    app.add_middleware(_AuthMiddleware, auth_token=auth_token)

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
