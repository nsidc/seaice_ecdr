"""Add Seki method Sea Ice Concentration thresholds to ancillary NetCDF files.

This script updates these files:

/share/apps/G02202_V6/v06r00_ancillary/G02202-ancillary-psn25-v06r00.nc
/share/apps/G02202_V6/v06r00_ancillary/G02202-ancillary-pss25-v06r00.nc

To include the data from the following CSV files:

/share/apps/G02202_V6/v06r00_ancillary/nh_final_thresholds.csv
/share/apps/G02202_V6/v06r00_ancillary/nh_final_thresholds-leap-year.csv
/share/apps/G02202_V6/v06r00_ancillary/sh_final_thresholds.csv
/share/apps/G02202_V6/v06r00_ancillary/sh_final_thresholds-leap-year.csv

As new variables:

* `cdr_conc_threshold`
* `cdr_conc_threshold_leap_year`

As as consequence of the addition of these variables, a new coordinate variable,
`doy` is added with 366 entries. The `cdr_conc_threshold` variable will have
it's 366th entry as np.nan
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pm_tb_data._types import Hemisphere

V6_ANC_DIR = Path("/share/apps/G02202_V6/v06r00_ancillary/")


def add_doy_variable(ds: xr.Dataset) -> xr.Dataset:
    """Add a Day of year (doy) variable to the dataset with 366 entries."""
    ds["doy"] = xr.DataArray(
        np.arange(1, 366 + 1, dtype=np.int16),
        dims=("doy"),
        attrs=dict(
            long_name="Day of year",
            comment="366 days are provided to account for leap years.",
        ),
    )

    return ds


def _read_thresholds_csv(hemisphere: Hemisphere):
    leap_threshold_csv_fp = (
        V6_ANC_DIR / f"{hemisphere[0].lower()}h_final_thresholds-leap-year.csv"
    )
    leap_thresholds_df = pd.read_csv(leap_threshold_csv_fp)

    return leap_thresholds_df


def add_threshold_variable(ds: xr.Dataset, hemisphere: Hemisphere) -> xr.Dataset:
    """Adds a threshold variable to the dataset, indexed by doy.

    This assumes that the threshold CSV contains thresholds as percentages.

    Stores the values as fractional values, as this is what the cdr_seaice_conc
    variable does.
    """
    leap_thresholds_df = _read_thresholds_csv(hemisphere)

    try:
        # Originally, we planned to apply the seki thresholding to DMSP
        # data. This variable may be an artifact of that. Drop if it exists.
        ds = ds.drop_vars(["dmsp_cdr_seaice_conc_threshold"])
        print("Dropped dmsp_cdr_seaice_conc_threshold")
    except Exception:
        pass

    ds["am2_cdr_seaice_conc_threshold"] = xr.DataArray(
        leap_thresholds_df.threshold.values / 100.0,
        dims=("doy"),
        attrs=dict(
            comment=(
                "Sea ice concentration threshold values applied to CDR for AMSR2 source data. "
                "Thresholds are derived from methods adapted from Seki et. al. 2024."
            ),
            units="1",
            long_name="CDR sea ice concentration threshold for AMSR2 data",
            valid_range=(np.uint8(0), np.uint8(100)),
        ),
    )

    # Set encoding for netcdf
    ds["am2_cdr_seaice_conc_threshold"].encoding = dict(
        zlib=True,
        dtype=np.uint8,
        scale_factor=0.01,
        add_offset=0.0,
        _FillValue=255,
    )

    return ds


def _read_ancillary_file(hemisphere: Hemisphere):
    ancillary_path = (
        V6_ANC_DIR / f"G02202-ancillary-ps{hemisphere[0].lower()}25-v06r00.nc"
    )
    ds = xr.open_dataset(ancillary_path, mask_and_scale=False)

    return ds


def process_thresholds_files(hemisphere: Hemisphere):
    ancillary_path = (
        V6_ANC_DIR / f"G02202-ancillary-ps{hemisphere[0].lower()}25-v06r00.nc"
    )
    ds = _read_ancillary_file(hemisphere)
    ds = add_doy_variable(ds)
    ds = add_threshold_variable(ds, hemisphere=hemisphere)
    ds.close()

    temp_filepath = V6_ANC_DIR / "anc_with_thresh.nc"
    ds.to_netcdf(temp_filepath, mode="w")

    shutil.move(ancillary_path, V6_ANC_DIR / (ancillary_path.name + ".bak"))
    shutil.move(temp_filepath, ancillary_path)


if __name__ == "__main__":
    for hemisphere in ("north", "south"):
        process_thresholds_files(hemisphere)

        # Test that we got what we wanted.
        ancillary_data = _read_ancillary_file(hemisphere)
        csv_df = _read_thresholds_csv(hemisphere)

        assert (
            csv_df.threshold.values
            == ancillary_data.am2_cdr_seaice_conc_threshold.values
        ).all()
