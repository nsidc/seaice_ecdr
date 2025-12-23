"""Code to regenerate checksums of CDR output files.

To run, manually update the `DATA_DIR` to the G02202/G10016 complete output
directory you want updated.
"""

from multiprocessing import Pool
from pathlib import Path

from pm_tb_data._types import Hemisphere

from seaice_ecdr.checksum import write_checksum_file

# DATA_DIR = Path("/share/apps/G10016_V4/v04r00/dev/trst2284/CDR/complete/")
# DATA_DIR = Path("/share/apps/G02202_V6/v06r00_outputs/dev/trst2284/complete")
# DATA_DIR = Path("/share/apps/G10016_V4/v04r00/production/CDR/complete")
DATA_DIR = Path("/share/apps/G02202_V6/v06r00_outputs/production/complete/")


def regenerate_for_hemisphere(hemisphere: Hemisphere):
    hemisphere_dir = DATA_DIR / hemisphere
    nc_fps = sorted(list(hemisphere_dir.rglob("*.nc")))

    # Create associated checksum file.
    for nc_fp in nc_fps:
        subdir = nc_fp.parent.relative_to(hemisphere_dir)
        checksum_dir = hemisphere_dir / "checksums" / subdir
        write_checksum_file(input_filepath=nc_fp, output_dir=checksum_dir)


if __name__ == "__main__":
    with Pool() as p:
        p.map(regenerate_for_hemisphere, ["north", "south"])
