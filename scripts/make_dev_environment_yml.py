"""Script to create `dev_environment.yml` from associated project environment.yml files.

Assumes `pm_icecon` and `pm_tb_data` repos are checked out in a directory
parallel to the `seaice_ecdr` repo.
"""

from pathlib import Path

import yaml
from loguru import logger

if __name__ == "__main__":
    deps = dict()
    this_dir = Path(__file__).parent
    for env_file in (
        this_dir / "../environment.yml",
        this_dir / "../../pm_icecon/environment.yml",
        this_dir / "../../pm_tb_data/environment.yml",
    ):
        with open(env_file, "r") as f:
            data = yaml.safe_load(f)
        for dep_and_pin in data["dependencies"]:
            result = dep_and_pin.split(" ")
            dep = result[0]

            # Skip direct dependencies on these other projects we're actively
            # working on.
            if dep in ("pm_icecon", "pm_tb_data"):
                continue

            if len(result) == 1:
                pin = ""
            else:
                pin = result[1]

            if dep not in deps:
                deps[dep] = pin

    deps_list = []
    for dep, pin in deps.items():
        if pin:
            deps_list.append(f"{dep} {pin}")
        else:
            deps_list.append(f"{dep}")

    env_struct = dict(
        name="seaice_ecdr",
        channels=[
            "conda-forge",
            "nsidc",
            "nodefaults",
        ],
        dependencies=deps_list,
    )

    dev_env_fp = (this_dir / ".." / "dev_environment.yml").resolve()
    with open(dev_env_fp, "w") as dev_env_file:
        yaml.safe_dump(env_struct, dev_env_file)

    logger.info(f"wrote {dev_env_fp}")
