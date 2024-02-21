import os
import sys

from loguru import logger

# Set the default minimum log notification to Warning
try:
    logger.remove(0)  # Removes previous logger info
except ValueError:
    logger.debug(f"Started logging in {__name__}")

env = os.environ.get("ENVIRONMENT")
_default_log_level = "DEBUG" if env == "dev" else "INFO"
logger.add(sys.stdout, level=_default_log_level)
