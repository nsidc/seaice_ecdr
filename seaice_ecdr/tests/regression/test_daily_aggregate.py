import datetime as dt
from pathlib import Path
from typing import Final

import datatree
import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr.daily_aggregate import (
    get_daily_aggregate_filepath,
    make_daily_aggregate_netcdf_for_year,
)
from seaice_ecdr.intermediate_daily import make_standard_cdecdr_netcdf
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
        daily_output_fp = publish_daily_nc(
            base_output_dir=base_output_dir,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
        )

        ds = datatree.open_datatree(daily_output_fp)
        datasets.append(ds)

    # Now generate the daily aggregate file from the three created above.
    make_daily_aggregate_netcdf_for_year(
        year=year,
        hemisphere=hemisphere,
        resolution=resolution,
        complete_output_dir=complete_output_dir,
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
    agg_root_ds = xr.open_dataset(aggregate_filepath)
    agg_cdr_supplementary_ds = xr.open_dataset(
        aggregate_filepath, group="cdr_supplementary"
    )
    # TODO: prototype_amsr2 does not currently exist in this regression test
    # because only data for the default platforms are being generated
    # agg_prototype_amsr2_ds = xr.open_dataset(aggregate_filepath, group="prototype_amsr2")

    checksum_filepath = (
        complete_output_dir
        / "checksums"
        / "aggregate"
        / (aggregate_filepath.name + ".mnf")
    )
    assert checksum_filepath.is_file()

    for idx, (day, daily_ds) in enumerate(zip(range(1, 3 + 1), datasets)):

        # First, check the root group.
        # TODO: once we can upgrade to xarray >2024.9, we should be able to use
        # the built-in xarray datatree integration, along with improvements that
        # allow dimensions to be inherited. This will let us directly compare
        # the `daily_ds` with the data returned from the netcdf file.

        # Select the current day from the aggregate dataset. This drops `time`
        # as a dim, so we re-expand and then remove from the `crs` variable,
        # which we expect to have no dimensions at all.
        selected_from_agg_root = agg_root_ds.sel(time=dt.datetime(year, 3, day))
        selected_from_agg_root = selected_from_agg_root.expand_dims("time")
        selected_from_agg_root["crs"] = selected_from_agg_root.crs.isel(
            time=0, drop=True
        )

        # We add the lat/lon fields to the aggregate dataset. We do not expect
        # them in the daily fields we're comaring to, so remove them.

        # Assert that both datasets are equal
        daily_ds_root = daily_ds.drop_nodes("cdr_supplementary").ds
        for var_name in daily_ds_root.variables:
            xr.testing.assert_allclose(
                selected_from_agg_root[var_name], daily_ds_root[var_name], atol=0.009
            )

        # Now, check the supplementary group

        # Select the current day from the aggregate dataset. This drops `time`
        # as a dim, so we re-expand.
        selected_from_agg_suppl = agg_cdr_supplementary_ds.sel(time=idx)
        selected_from_agg_suppl = selected_from_agg_suppl.expand_dims("time")

        # Assert that both datasets are equal
        daily_ds_suppl = daily_ds["cdr_supplementary"].ds
        for var_name in daily_ds_suppl.variables:
            xr.testing.assert_allclose(
                selected_from_agg_suppl[var_name], daily_ds_suppl[var_name], atol=0.009
            )
