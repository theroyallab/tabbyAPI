import asyncio
from typing import Optional
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from common import config
from common.logger import UVICORN_LOG_CONFIG
from common.networking import get_global_depends
from common.utils import unwrap
from endpoints.Kobold import router as KoboldRouter
from endpoints.OAI import router as OAIRouter
from endpoints.core.router import router as CoreRouter


def setup_app(host: Optional[str] = None, port: Optional[int] = None):
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
    router_mapping = {"oai": OAIRouter, "kobold": KoboldRouter}

    # Include the OAI api by default
    if api_servers:
        for server in api_servers:
            selected_server = router_mapping.get(server.lower())

            if selected_server:
                app.include_router(selected_server.setup())

                logger.info(f"Starting {selected_server.api_name} API")
                for path, url in selected_server.urls.items():
                    formatted_url = url.format(host=host, port=port)
                    logger.info(f"{path}: {formatted_url}")
    else:
        app.include_router(OAIRouter.setup())
        for path, url in OAIRouter.urls.items():
            formatted_url = url.format(host=host, port=port)
            logger.info(f"{path}: {formatted_url}")

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
    # logger.info(f"Completions: http://{host}:{port}/v1/completions")
    # logger.info(f"Chat completions: http://{host}:{port}/v1/chat/completions")

    # Setup app
    app = setup_app(host, port)

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
