import hashlib
from pathlib import Path

from loguru import logger

from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR


def _get_checksum(filepath):
    with filepath.open("rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def write_checksum_file(*, input_filepath: Path, output_dir: Path):
    checksum = _get_checksum(input_filepath)

    size_in_bytes = input_filepath.stat().st_size
    filename = input_filepath.name

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / (filename + ".mnf")

    with open(output_filepath, "w") as checksum_file:
        checksum_file.write(f"{filename},{checksum},{size_in_bytes}")

    logger.info(f"Wrote checksum file {output_filepath}")


def get_checksum_dir(*, ecdr_data_dir: Path):
    checksum_dir = ecdr_data_dir / "checksums"

    return checksum_dir


if __name__ == "__main__":
    ecdr_data_dir = STANDARD_BASE_OUTPUT_DIR
    checksum_dir = ecdr_data_dir / "checksums"
    checksum_dir.mkdir(exist_ok=True)

    for product in ("complete_daily", "monthly", "aggregate"):
        input_dir = ecdr_data_dir / product
        output_dir = checksum_dir / product
        output_dir.mkdir(exist_ok=True)
        for nc_filepath in input_dir.glob("*.nc"):
            write_checksum_file(input_filepath=nc_filepath, output_dir=output_dir)
