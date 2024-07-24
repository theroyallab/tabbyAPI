import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from common import config
from common.logger import UVICORN_LOG_CONFIG
from common.networking import get_global_depends
from common.utils import unwrap
from endpoints.core.router import router as CoreRouter
from endpoints.OAI.router import router as OAIRouter


def setup_app():
    """Includes the correct routers for startup"""

    app = FastAPI(
        title="TabbyAPI",
        summary="An OAI compatible exllamav2 API that's both lightweight and fast",
        description=(
            "This docs page is not meant to send requests! Please use a service "
            "like Postman or a frontend UI."
        ),
        dependencies=get_global_depends(),
    )

    # ALlow CORS requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_servers = unwrap(config.network_config().get("api_servers"), [])

    # Map for API id to server router
    router_mapping = {"oai": OAIRouter}

    # Include the OAI api by default
    if api_servers:
        for server in api_servers:
            server_name = server.lower()
            if server_name in router_mapping:
                app.include_router(router_mapping[server_name])
    else:
        app.include_router(OAIRouter)

    # Include core API request paths
    app.include_router(CoreRouter)

    return app


def export_openapi():
    """Function to return the OpenAPI JSON from the API server"""

    app = setup_app()
    return app.openapi()


async def start_api(host: str, port: int):
    """Isolated function to start the API server"""

    # TODO: Move OAI API to a separate folder
    logger.info(f"Developer documentation: http://{host}:{port}/redoc")
    logger.info(f"Completions: http://{host}:{port}/v1/completions")
    logger.info(f"Chat completions: http://{host}:{port}/v1/chat/completions")

    # Setup app
    app = setup_app()

    # Get the current event loop
    loop = asyncio.get_running_loop()

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_config=UVICORN_LOG_CONFIG,
        loop=loop,
    )
    server = uvicorn.Server(config)

    await server.serve()
