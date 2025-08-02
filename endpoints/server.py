import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import torch
import torch.distributed as dist
from typing import Optional

from common.logger import UVICORN_LOG_CONFIG
from common.networking import get_global_depends
from common.tabby_config import config
from endpoints.Kobold import router as KoboldRouter
from endpoints.OAI import router as OAIRouter
from endpoints.core.router import router as CoreRouter


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    torch.cuda.synchronize()
    dist.destroy_process_group()


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
        lifespan=lifespan,
    )

    # ALlow CORS requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
    for server in api_servers:
        selected_server = router_mapping.get(server.lower())

        if selected_server:
            app.include_router(selected_server.setup())

            logger.info(f"Starting {selected_server.api_name} API")
            for path, url in selected_server.urls.items():
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
