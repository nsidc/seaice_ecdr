import sys

from loguru import logger

# Set the default minimum log notification to Warning
try:
    logger.remove(0)  # Removes previous logger info
except ValueError:
    logger.debug(f"Started logging in {__name__}")

logger.add(sys.stderr, level="WARNING")
