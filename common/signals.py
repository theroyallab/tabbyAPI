import signal
import sys
from loguru import logger
from types import FrameType


def signal_handler(*_):
    """Signal handler for main function. Run before uvicorn starts."""

    logger.warning("Shutdown signal called. Exiting gracefully.")
    sys.exit(0)


def uvicorn_signal_handler(signal_event: signal.Signals):
    """Overrides uvicorn's signal handler."""

    default_signal_handler = signal.getsignal(signal_event)

    def wrapped_handler(signum: int, frame: FrameType = None):
        logger.warning("Shutdown signal called. Exiting gracefully.")
        default_signal_handler(signum, frame)

    signal.signal(signal_event, wrapped_handler)
