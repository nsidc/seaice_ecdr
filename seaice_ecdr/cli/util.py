"""util.py  cli routines common to seaice_ecdr."""

import datetime as dt
import subprocess
from pathlib import Path

_THIS_DIR = Path(__file__).parent

CLI_EXE_PATH = (_THIS_DIR / "../../scripts/cli.sh").resolve()


def datetime_to_date(_ctx, _param, value: dt.datetime) -> dt.date:
    """Click callback that takes a `dt.datetime` and returns `dt.date`."""
    return value.date()


def run_cmd(cmd: str) -> None:
    """Runs the given command in a shell-enabled subprocess."""
    subprocess.run(
        cmd,
        shell=True,
        check=True,
    )
