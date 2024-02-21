import os
import sys

from loguru import logger

env = os.environ.get("ENVIRONMENT")
_default_log_level = "DEBUG" if env == "dev" else "INFO"

logger.configure(
    handlers=[
        dict(sink=sys.stdout, level=_default_log_level),
    ]
)
