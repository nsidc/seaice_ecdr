"""Script to update DOIs for G02202 V6 and G10116 V4.

On Dec. 11, 2025, we realized that the DOI for these products was wrong (used
the previous version). This script updates existing nc files to have the correct
DOI in order to avoid data reprocessing.
"""

from pathlib import Path

from netCDF4 import Dataset

FINAL_DATA_DIR = Path("/share/apps/G02202_V6/v06r00_outputs/dev/trst2284/complete")
NRT_DATA_DIR = Path("/share/apps/G10016_V4/v04r00/dev/trst2284/CDR/complete/")

FINAL_DOI = "https://doi.org/10.7265/b18j-z797"
NRT_DOI = "https://doi.org/10.7265/tm2n-1m33"


def update_dois(data_dir, doi):
    print(f"Updating nc files in {data_dir} with doi {doi}")
    for nc_file in data_dir.rglob("*.nc"):
        with Dataset(str(nc_file), "a") as nc:
            if "id" in nc.ncattrs():
                nc.setncattr("id", doi)
            else:
                print(
                    f"Did not update doi in {nc_file} because it lacked an `id` attr."
                )


if __name__ == "__main__":
    update_dois(FINAL_DATA_DIR, FINAL_DOI)
    update_dois(NRT_DATA_DIR, NRT_DOI)
