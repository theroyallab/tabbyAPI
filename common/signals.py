import sys
from common.logger import xlogger

SHUTTING_DOWN: bool = False

# Set by start_api before uvicorn takes over signal handling.
# Uvicorn captures SIGINT/SIGTERM while serving and re-raises them after its
# graceful shutdown, which lands in signal_handler below.
SERVER_SERVING: bool = False


def signal_handler(*_):
    """
    Signal handler for the main function. Runs while uvicorn isn't serving:
    before startup completes, and again if uvicorn re-raises a captured
    signal after its graceful shutdown.
    """

    global SHUTTING_DOWN

    if SHUTTING_DOWN:
        return

    xlogger.warning("Shutdown signal called. Exiting gracefully.")
    SHUTTING_DOWN = True

    # If uvicorn already served and shut down, return so entrypoint_async
    # can unload models in normal control flow.
    if SERVER_SERVING:
        return

    # Otherwise (startup, or a synchronous context), abort the process.
    # A partially loaded model is reclaimed on process exit.
    sys.exit(0)
