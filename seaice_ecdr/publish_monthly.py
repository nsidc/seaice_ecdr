from pathlib import Path


def get_complete_monthly_dir(complete_output_dir: Path) -> Path:
    monthly_dir = complete_output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    return monthly_dir
