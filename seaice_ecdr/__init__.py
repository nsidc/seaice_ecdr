import datetime as dt
import os
import sys

from loguru import logger

from seaice_ecdr.constants import LOGS_DIR

__version__ = "v0.1.0"

DEFAULT_LOG_LEVEL = "INFO"

# If we're in dev, DEBUG level logs are appropriate, otherwise use INFO.
env = os.environ.get("ENVIRONMENT")
_default_log_level = "DEBUG" if env == "dev" else DEFAULT_LOG_LEVEL

# Configure the default logger, which prints to stderr.
logger.configure(
    handlers=[
        dict(sink=sys.stderr, level=_default_log_level),
    ],
)

# Optionally (by default) log to a file named after the current date.
# TODO: consider logging all messages (TRACE level)? We could retain all context for
# regular ops runs up to a certain date, which might help w/ debugging
# issues...Larger reprocessing efforts could disable file logging (or change the
# level?) for speed and space consideration issues.
disable_file_logging = os.environ.get("DISABLE_FILE_LOGGING")
do_not_log = disable_file_logging is not None and disable_file_logging.upper() in (
    "TRUE",
    "YES",
)

if not do_not_log:
    # One file per day.
    LOGS_DIR.mkdir(exist_ok=True)
    # file_sink_fp = LOGS_DIR / f"{time:%Y-%m-%d}.log"
    file_sink_fp = LOGS_DIR / f"{dt.datetime.now():%Y-%m-%d}.log"
    logger.debug(f"Logging to {file_sink_fp}")
    logger.add(
        file_sink_fp,
        level=DEFAULT_LOG_LEVEL,
        # Retain logs for up to a month.
        retention=31,
    )
else:
    logger.debug(f"Not logging to file (DISABLE_FILE_LOGGING={disable_file_logging}).")
