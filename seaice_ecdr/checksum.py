"""Code for writing checksum files for complete ECDR output nc files.

This module can be run as a script to (re-)generate checksum files for all
existing standard (not NRT) output files.
"""

import hashlib
from pathlib import Path

from loguru import logger


def _get_checksum(filepath):
    with filepath.open("rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def write_checksum_file(*, input_filepath: Path, output_dir: Path) -> Path:
    checksum = _get_checksum(input_filepath)

    size_in_bytes = input_filepath.stat().st_size

    checksum_filename = input_filepath.name + ".mnf"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / checksum_filename
    with open(output_filepath, "w") as checksum_file:
        checksum_file.write(f"{input_filepath.name},{checksum},{size_in_bytes}")

    logger.success(f"Wrote checksum file {output_filepath}")

    return output_filepath
