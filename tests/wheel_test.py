""" Test if the wheels are installed correctly. """
from importlib.metadata import version
from importlib.util import find_spec

from logger import init_logger

logger = init_logger(__name__)

successful_packages = []
errored_packages = []

if find_spec("flash_attn") is not None:
    logger.info(
        f"Flash attention on version {version('flash_attn')} " "successfully imported"
    )
    successful_packages.append("flash_attn")
else:
    logger.error("Flash attention 2 is not found in your environment.")
    errored_packages.append("flash_attn")

if find_spec("exllamav2") is not None:
    logger.info(f"Exllamav2 on version {version('exllamav2')} " "successfully imported")
    successful_packages.append("exllamav2")
else:
    logger.error("Exllamav2 is not found in your environment.")
    errored_packages.append("exllamav2")

if find_spec("torch") is not None:
    logger.info(f"Torch on version {version('torch')} successfully imported")
    successful_packages.append("torch")
else:
    logger.error("Torch is not found in your environment.")
    errored_packages.append("torch")

if find_spec("jinja2") is not None:
    logger.info(f"Jinja2 on version {version('jinja2')} successfully imported")
    successful_packages.append("jinja2")
else:
    logger.error("Jinja2 is not found in your environment.")
    errored_packages.append("jinja2")

logger.info(f"\nSuccessful imports: {', '.join(successful_packages)}")
logger.error(f"Errored imports: {''.join(errored_packages)}")

if len(errored_packages) > 0:
    logger.warning(
        "If all packages are installed, but not found "
        "on this test, please check the wheel versions for the "
        "correct python version and CUDA version (if "
        "applicable)."
    )
else:
    logger.info("All wheels are installed correctly.")
