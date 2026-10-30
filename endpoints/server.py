import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from typing import Optional
from urllib.parse import quote

from common import auth, signals
from common.logger import UVICORN_LOG_CONFIG
from common.errors import (
    ContextLengthHTTPException,
    LoopDetectedHTTPException,
    context_length_exception_handler,
    loop_detected_exception_handler,
)
from common.networking import get_global_depends
from common.optional_dependencies import package_version_info
from common.tabby_config import config
from endpoints.Kobold import router as KoboldRouter
from endpoints.OAI import router as OAIRouter
from endpoints.core.router import router as CoreRouter
from endpoints.ui.router import mount_ui


def setup_app(host: Optional[str] = None, port: Optional[int] = None):
    """Includes the correct routers for startup"""

    # The status page reports the backend's version and commit. Read now, so a
    # source checkout updated while the server runs doesn't misreport it
    package_version_info("exllamav3")

    app = FastAPI(
        title="TabbyAPI",
        summary="An OAI compatible exllamav3 API that's both lightweight and fast",
        description=(
            "This docs page is not meant to send requests! Please use a service "
            "like Postman or a frontend UI."
        ),
        dependencies=get_global_depends(),
    )
    app.add_exception_handler(ContextLengthHTTPException, context_length_exception_handler)
    app.add_exception_handler(LoopDetectedHTTPException, loop_detected_exception_handler)

    # Allow CORS requests from the configured origins.
    # allow_credentials stays False: TabbyAPI authenticates with a header/query
    # token rather than cookies, so credentialed CORS buys nothing and would make
    # Starlette reflect an arbitrary requesting origin back instead of sending "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.network.allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_servers = config.network.api_servers
    api_servers = (
        api_servers
        if api_servers
        else [
            "oai",
        ]
    )

    # Map for API id to server router
    router_mapping = {"oai": OAIRouter, "kobold": KoboldRouter}

    # Include the OAI api by default
    enabled_apis = []
    for server in api_servers:
        selected_server = router_mapping.get(server.lower())

        if selected_server:
            app.include_router(selected_server.setup())
            enabled_apis.append(selected_server.api_name)

    if host is not None:
        logger.info(
            f"Serving {', '.join(enabled_apis)} API on http://{host}:{port} "
            f"(docs at http://{host}:{port}/redoc)"
        )

    # Include core API request paths
    app.include_router(CoreRouter)

    if config.network.web_ui:
        mount_ui(app)

        if host is not None:
            # Log in with a click: the admin key is printed at startup anyway
            # (unless log_auth_keys is off), and a URL fragment never reaches
            # the server or its logs
            fragment = ""
            if not auth.DISABLE_AUTH and auth.AUTH_KEYS and auth.log_auth_keys():
                fragment = "#key=" + quote(auth.AUTH_KEYS.admin_key, safe="")
            logger.info(f"Chat client at http://{host}:{port}/ui/{fragment}")
            logger.info(
                f"Server status and model loading at http://{host}:{port}/ui/status{fragment}"
            )

    return app


def export_openapi():
    """Function to return the OpenAPI JSON from the API server"""

    app = setup_app()
    return app.openapi()


async def start_api(host: str, port: int):
    """Isolated function to start the API server"""

    # Setup app
    app = setup_app(host, port)

    # Get the current event loop
    loop = asyncio.get_running_loop()

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=UVICORN_LOG_CONFIG,
        access_log=config.network.access_log,
        loop=loop,
    )
    server = uvicorn.Server(uvicorn_config)

    # Uvicorn owns SIGINT/SIGTERM while serving and re-raises captured
    # signals after its graceful shutdown
    signals.SERVER_SERVING = True

    await server.serve()
