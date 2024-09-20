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
    # If an event loop doesn't exist (synchronous), exit.
    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(signal_handler_async())
    except RuntimeError:
        sys.exit(0)


async def signal_handler_async():
    """
    Internal signal handler. Runs all async code to shut down the program.

    asyncio.run will cancel all remaining tasks and close the event loop.
    """

    if model.container:
        await model.unload_model(skip_wait=True, shutdown=True)

    if model.embeddings_container:
        await model.unload_embedding_model()


def uvicorn_signal_handler(signal_event: signal.Signals):
    """Overrides uvicorn's signal handler."""

    default_signal_handler = signal.getsignal(signal_event)

    def wrapped_handler(signum: int, frame: FrameType = None):
        logger.warning("Shutdown signal called. Exiting gracefully.")
        default_signal_handler(signum, frame)

    signal.signal(signal_event, wrapped_handler)
