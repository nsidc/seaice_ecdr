"""Code for writing checksum files for complete ECDR output nc files.

This module can be run as a script to (re-)generate checksum files for all
existing standard (not NRT) output files.
"""

import hashlib
from pathlib import Path
from typing import Literal

from loguru import logger


def _get_checksum(filepath):
    with filepath.open("rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def get_checksum_filepath(
    *,
    input_filepath: Path,
    ecdr_data_dir: Path,
    product_type: Literal["daily", "monthly", "aggregate"],
) -> Path:
    checksum_dir = get_checksum_dir(ecdr_data_dir=ecdr_data_dir)
    checksum_product_dir = checksum_dir / product_type
    checksum_product_dir.mkdir(exist_ok=True)

    checksum_filename = input_filepath.name + ".mnf"
    checksum_dir = get_checksum_dir(ecdr_data_dir=ecdr_data_dir)
    checksum_filepath = checksum_dir / product_type / checksum_filename

    return checksum_filepath


def write_checksum_file(
    *,
    input_filepath: Path,
    ecdr_data_dir: Path,
    product_type: Literal["daily", "monthly", "aggregate"],
) -> Path:
    checksum = _get_checksum(input_filepath)

    size_in_bytes = input_filepath.stat().st_size
    output_filepath = get_checksum_filepath(
        input_filepath=input_filepath,
        ecdr_data_dir=ecdr_data_dir,
        product_type=product_type,
    )

    with open(output_filepath, "w") as checksum_file:
        checksum_file.write(f"{input_filepath.name},{checksum},{size_in_bytes}")

    logger.success(f"Wrote checksum file {output_filepath}")

    return output_filepath


def get_checksum_dir(*, ecdr_data_dir: Path):
    checksum_dir = ecdr_data_dir / "checksums"
    checksum_dir.mkdir(exist_ok=True)

    return checksum_dir
