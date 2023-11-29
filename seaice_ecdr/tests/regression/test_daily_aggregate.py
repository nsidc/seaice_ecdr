import datetime as dt
from pathlib import Path
from typing import Final

import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr.complete_daily_ecdr import read_or_create_and_read_cdecdr_ds
from seaice_ecdr.daily_aggregate import (
    get_daily_aggregate_filepath,
    make_daily_aggregate_netcdf_for_year,
)


def test_daily_aggreagate_matches_daily_data(tmpdir):
    tmpdir_path = Path(tmpdir)

    year = 2022
    resolution: Final = "12.5"
    hemisphere: Final = NORTH
    sat: Final = "am2"

    # First, ensure some daily data is created.
    datasets = []
    for day in range(1, 3 + 1):
        ds = read_or_create_and_read_cdecdr_ds(
            date=dt.date(year, 3, day),
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=tmpdir_path,
        )
        datasets.append(ds)

    # Now generate the daily aggregate file from the three created above.
    make_daily_aggregate_netcdf_for_year(
        year=year,
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        ecdr_data_dir=tmpdir_path,
    )

    # Read back in the data.
    aggregate_filepath = get_daily_aggregate_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        ecdr_data_dir=tmpdir_path,
        start_date=dt.date(year, 3, 1),
        end_date=dt.date(year, 3, 3),
    )

    # Assert that there are data for each of the three expected dates, and that
    # they match the daily final datasets.
    agg_ds = xr.open_dataset(aggregate_filepath)

    for day, daily_ds in zip(range(1, 3 + 1), datasets):
        # Select the current day from the aggregate dataset. This drops `time`
        # as a dim, so we re-expand and then remove from the `crs` variable,
        # which we expect to have no dimensions at all.
        selected_from_agg = agg_ds.sel(time=dt.datetime(year, 3, day))
        selected_from_agg = selected_from_agg.expand_dims("time")
        selected_from_agg["crs"] = selected_from_agg.crs.isel(time=0, drop=True)

        # Assert that both datasets are equal
        xr.testing.assert_allclose(selected_from_agg, daily_ds, atol=0.009)
