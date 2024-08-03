import asyncio
import signal
import sys
from loguru import logger
from types import FrameType

from common import model


SHUTTING_DOWN: bool = False


def signal_handler(*_):
    """Signal handler for main function. Run before uvicorn starts."""

    global SHUTTING_DOWN

    if SHUTTING_DOWN:
        return

    logger.warning("Shutdown signal called. Exiting gracefully.")
    SHUTTING_DOWN = True

    # Run async unloads for model
    asyncio.ensure_future(signal_handler_async())


async def signal_handler_async(*_):
    """Internal signal handler. Runs all async code to shut down the program."""

    if model.container:
        await model.unload_model(skip_wait=True, shutdown=True)

    if model.embeddings_container:
        await model.unload_embedding_model()

    # Exit the program
    sys.exit(0)


def uvicorn_signal_handler(signal_event: signal.Signals):
    """Overrides uvicorn's signal handler."""

    default_signal_handler = signal.getsignal(signal_event)

    def wrapped_handler(signum: int, frame: FrameType = None):
        logger.warning("Shutdown signal called. Exiting gracefully.")
        default_signal_handler(signum, frame)

    signal.signal(signal_event, wrapped_handler)
