import datetime as dt
from pathlib import Path
from typing import Final

import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr.complete_daily_ecdr import make_standard_cdecdr_netcdf, read_cdecdr_ds
from seaice_ecdr.daily_aggregate import (
    get_daily_aggregate_filepath,
    make_daily_aggregate_netcdf_for_year,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.publish_daily import publish_daily_nc
from seaice_ecdr.util import get_complete_output_dir


def test_daily_aggregate_matches_daily_data(tmpdir):
    base_output_dir = Path(tmpdir)
    hemisphere: Final = NORTH
    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )

    year = 2022
    resolution: Final = "25"
    land_spillover_alg: Final = "NT2"
    ancillary_source: Final = "CDRv5"

    # First, ensure some daily data is created.
    datasets = []
    for day in range(1, 3 + 1):
        date = dt.date(year, 3, day)
        make_standard_cdecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            base_output_dir=base_output_dir,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )
        publish_daily_nc(
            base_output_dir=base_output_dir,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
        )

        # TODO: this fucntin is really inteded to just read the ds associated
        # with the intermediate file, but it can (I think) be used for the
        # published version...
        platform = PLATFORM_CONFIG.get_platform_by_date(date)
        ds = read_cdecdr_ds(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=complete_output_dir,
            platform_id=platform.id,
            is_nrt=False,
        )
        datasets.append(ds)

    # Now generate the daily aggregate file from the three created above.
    make_daily_aggregate_netcdf_for_year(
        year=year,
        hemisphere=hemisphere,
        resolution=resolution,
        complete_output_dir=complete_output_dir,
        ancillary_source=ancillary_source,
    )

    # Read back in the data.
    aggregate_filepath = get_daily_aggregate_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        complete_output_dir=complete_output_dir,
        start_date=dt.date(year, 3, 1),
        end_date=dt.date(year, 3, 3),
    )

    # Assert that there are data for each of the three expected dates, and that
    # they match the daily final datasets.
    agg_ds = xr.open_dataset(aggregate_filepath)

    checksum_filepath = (
        complete_output_dir
        / "checksums"
        / "aggregate"
        / (aggregate_filepath.name + ".mnf")
    )
    assert checksum_filepath.is_file()

    for day, daily_ds in zip(range(1, 3 + 1), datasets):
        # Select the current day from the aggregate dataset. This drops `time`
        # as a dim, so we re-expand and then remove from the `crs` variable,
        # which we expect to have no dimensions at all.
        selected_from_agg = agg_ds.sel(time=dt.datetime(year, 3, day))
        selected_from_agg = selected_from_agg.expand_dims("time")
        selected_from_agg["crs"] = selected_from_agg.crs.isel(time=0, drop=True)

        # We add the lat/lon fields to the aggregate dataset. We do not expect
        # them in the daily fields we're comaring to, so remove them.
        selected_from_agg = selected_from_agg.drop_vars(["latitude", "longitude"])

        # Assert that both datasets are equal
        xr.testing.assert_allclose(selected_from_agg, daily_ds, atol=0.009)
