import hashlib
from pathlib import Path


def _get_checksum(filepath):
    with filepath.open("rb") as file:
        return hashlib.md5(file.read()).hexdigest()


def write_checksum_file(*, input_filepath: Path, output_dir: Path):
    checksum = _get_checksum(input_filepath)

    size_in_bytes = input_filepath.stat().st_size
    filename = input_filepath.name

    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / (filename + ".mnf")

    with open(output_filepath, "w") as checksum_file:
        checksum_file.write(f"{filename},{checksum},{size_in_bytes}")

    print(f"Wrote checksum file {output_filepath}")
