"""Create the initial daily ECDR file.

Initially based on pm_icecon's 'pm_cdr.py' routines
"""

import datetime as dt
import sys
import traceback
from pathlib import Path
from typing import get_args

import click
import numpy as np
import numpy.typing as npt
import pm_icecon.bt.bt_params as pmi_bt_params
import pm_icecon.bt.compute_bt_ic as bt
import pm_icecon.bt.params.amsr2 as bt_amsr2_params
import pm_icecon.nt.compute_nt_ic as nt
import pm_icecon.nt.params.amsr2 as nt_amsr2_params
import xarray as xr
from loguru import logger
from seaice_ecdr.pm_cdr import cdr
from pm_icecon._types import Hemisphere
from pm_icecon.cli.util import datetime_to_date
from pm_icecon.config.models.bt import BootstrapParams
from pm_icecon.constants import CDR_DATA_DIR, DEFAULT_FLAG_VALUES
from pm_icecon.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_icecon.util import date_range, standard_output_filename

from seaice_ecdr.gridid_to_xr_dataarray import get_dataset_for_gridid


def cdr_bootstrap(
    date: dt.date,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_params: BootstrapParams,
    bt_tb_mask,
    bt_weather_mask,
):
    """Generate the raw bootstrap concentration field."""
    bt_conc = bt.bootstrap_for_cdr(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        params=bt_params,
        tb_mask=bt_tb_mask,
        weather_mask=bt_weather_mask,
    )

    return bt_conc


def cdr_nasateam(
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_v19: npt.NDArray,
    nt_tiepoints: NasateamTiePoints,
) -> npt.NDArray:
    """Generate the raw NASA Team concentration field."""
    # Compute the NASA Team conc field
    # Note that concentrations from nasateam may be >100%
    nt_pr_1919 = nt.compute_ratio(tb_v19, tb_h19)
    nt_gr_3719 = nt.compute_ratio(tb_v37, tb_v19)
    nt_conc = nt.calc_nasateam_conc(
        pr_1919=nt_pr_1919,
        gr_3719=nt_gr_3719,
        tiepoints=nt_tiepoints,
    )

    return nt_conc


def get_bt_tb_mask(
    tb_v37,
    tb_h37,
    tb_v19,
    tb_v22,
    mintb,
    maxtb,
    tb_data_mask_function,
):
    """Determine TB mask per Bootstrap algorithm's criteria."""
    bt_tb_mask = tb_data_mask_function(
        tbs=(
            tb_v37,
            tb_h37,
            tb_v19,
            tb_v22,
        ),
        min_tb=mintb,
        max_tb=maxtb,
    )

    try:
        assert tb_v37.shape == tb_h37.shape
        assert tb_v37.shape == tb_v22.shape
        assert tb_v37.shape == tb_v19.shape
        # assert tb_v37.shape == bt_params.land_mask.shape
        assert tb_v37.shape == bt_tb_mask.shape
    except AssertionError as e:
        raise ValueError(f'Mismatched shape error in get_bt_tb_mask\n{e}')

    return bt_tb_mask


def calculate_cdr_conc(
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_params: BootstrapParams,
    nt_tiepoints: NasateamTiePoints,
    nt_gradient_thresholds: NasateamGradientRatioThresholds,
    nt_invalid_ice_mask: npt.NDArray[np.bool_],
    nt_minic: npt.NDArray,
    nt_shoremap: npt.NDArray,
    missing_flag_value,
) -> npt.NDArray:
    """Run the CDR algorithm."""
    # First, get bootstrap conc.
    bt_tb_mask = get_bt_tb_mask(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        tb_v22=tb_v22,
        mintb=bt_params.mintb,
        maxtb=bt_params.maxtb,
        tb_data_mask_function=bt.tb_data_mask,
    )

    bt_weather_mask = bt.get_weather_mask(
        v37=tb_v37,
        h37=tb_h37,
        v22=tb_v22,
        v19=tb_v19,
        land_mask=bt_params.land_mask,
        tb_mask=bt_tb_mask,
        ln1=bt_params.vh37_params.lnline,
        date=date,
        weather_filter_seasons=bt_params.weather_filter_seasons,
    )

    # bt_invalid_ice_mask = bt_params.invalid_ice_mask

    bt_conc = cdr_bootstrap(
        date,
        tb_v37,
        tb_h37,
        tb_v19,
        tb_v22,
        bt_params,
        bt_tb_mask,
        bt_weather_mask,
    )

    # Next, get nasateam conc. Note that concentrations from nasateam may be
    # >100%.
    nt_conc = cdr_nasateam(
        date,
        tb_h19,
        tb_v37,
        tb_v19,
        nt_tiepoints,
    )

    # Now calculate CDR SIC
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice
    # Note: Here, values without sea ice (because no TBs) have val np.nan
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    # Apply masks
    # Get Nasateam weather filter
    nt_gr_2219 = nt.compute_ratio(tb_v22, tb_v19)
    nt_gr_3719 = nt.compute_ratio(tb_v37, tb_v19)
    nt_weather_mask = nt.get_weather_filter_mask(
        gr_2219=nt_gr_2219,
        gr_3719=nt_gr_3719,
        gr_2219_threshold=nt_gradient_thresholds['2219'],
        gr_3719_threshold=nt_gradient_thresholds['3719'],
    )
    # Apply weather filters and invalid ice masks
    # TODO: can we just use a single invalid ice mask?
    # Note: We do not want to set zero sic where we have no TBs
    set_to_zero_sic = (
        nt_weather_mask
        | bt_weather_mask
        | nt_invalid_ice_mask
        | bt_params.invalid_ice_mask
    )
    cdr_conc[set_to_zero_sic] = 0

    # Apply land spillover corrections
    # TODO: eventually, we want each of these routines to return a e.g.,
    #   delta that can be applied to the input concentration
    #   instead of returning a new conc. Then we would have a
    #   seprate algorithm for choosing how to apply
    #   multiple spillover deltas to a given conc field.

    # nasateam first:
    cdr_conc = nt.apply_nt_spillover(
        conc=cdr_conc,
        shoremap=nt_shoremap,
        minic=nt_minic,
    )
    # then bootstrap:
    cdr_conc = bt.coastal_fix(
        conc=cdr_conc,
        missing_flag_value=missing_flag_value,
        land_mask=bt_params.land_mask,
        minic=bt_params.minic,
    )

    # Apply land flag value and clamp max conc to 100.
    # TODO: extract this func from nt and allow override of flag values
    cdr_conc = nt._clamp_conc_and_set_flags(
        shoremap=nt_shoremap,
        conc=cdr_conc,
    )

    # Return CDR.
    # TODO: return an xr dataset with variables containing the outputs of
    # intermediate steps above.
    return cdr_conc


def amsr2_cdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Create a CDR-like concentration field from AMSR2 data."""
    # Get AMSR2 TBs
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    bt_params = bt_amsr2_params.get_amsr2_params(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    nt_params = nt_amsr2_params.get_amsr2_params(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # finally, compute the CDR.
    conc = cdr(
        date=date,
        tb_h19=spatial_interp_tbs(xr_tbs['h18'].data),
        tb_v37=spatial_interp_tbs(xr_tbs['v36'].data),
        tb_h37=spatial_interp_tbs(xr_tbs['h36'].data),
        tb_v19=spatial_interp_tbs(xr_tbs['v18'].data),
        tb_v22=spatial_interp_tbs(xr_tbs['v23'].data),
        bt_params=bt_params,
        nt_tiepoints=nt_params.tiepoints,
        nt_gradient_thresholds=nt_params.gradient_thresholds,
        # TODO: this is the same as the bootstrap mask!
        nt_invalid_ice_mask=bt_params.invalid_ice_mask,
        nt_minic=nt_params.minic,
        nt_shoremap=nt_params.shoremap,
        missing_flag_value=DEFAULT_FLAG_VALUES.missing,
    )

    cdr_conc_ds = xr.Dataset({'conc': (('y', 'x'), conc)})

    return cdr_conc_ds


def compute_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Create xr dataset containing the first pass of daily enhanced CDR."""
    # Note: at first, this is simply a copy of amsr2_cdr

    # Set the gridid
    if hemisphere == 'north' and resolution == '12':
        gridid = 'psn12.5'
    elif hemisphere == 'south' and resolution == '12':
        gridid = 'pss12.5'
    else:
        raise RuntimeError(
            f'Could not determine gridid from:\n' f'{hemisphere} and {resolution}'
        )

    # Initialize geo-referenced xarray Dataset
    ecdr_conc_ds = get_dataset_for_gridid(gridid, date)

    # Get AMSR2 TBs
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    """
    bt_params_orig = bt_amsr2_params.get_amsr2_params(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    """

    bt_params = pmi_bt_params.get_bootstrap_params(
        date=date,
        satellite='amsr2',
        gridid=gridid,
    )
    bt_fields = pmi_bt_params.get_bootstrap_fields(
        date=date,
        satellite='amsr2',
        gridid=gridid,
    )
    pmicecon_bt_params = pmi_bt_params.convert_to_pmicecon_bt_params(
        hemisphere, bt_params, bt_fields
    )

    nt_params = nt_amsr2_params.get_amsr2_params(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # finally, compute the CDR.
    conc = calculate_cdr_conc(
        date=date,
        tb_h19=spatial_interp_tbs(xr_tbs['h18'].data),
        tb_v37=spatial_interp_tbs(xr_tbs['v36'].data),
        tb_h37=spatial_interp_tbs(xr_tbs['h36'].data),
        tb_v19=spatial_interp_tbs(xr_tbs['v18'].data),
        tb_v22=spatial_interp_tbs(xr_tbs['v23'].data),
        bt_params=pmicecon_bt_params,
        nt_tiepoints=nt_params.tiepoints,
        nt_gradient_thresholds=nt_params.gradient_thresholds,
        # TODO: this is the same as the bootstrap mask!
        nt_invalid_ice_mask=pmicecon_bt_params.invalid_ice_mask,
        nt_minic=nt_params.minic,
        nt_shoremap=nt_params.shoremap,
        missing_flag_value=DEFAULT_FLAG_VALUES.missing,
    )

    ecdr_conc_ds['conc'] = (
        ('time', 'y', 'x'),
        np.expand_dims(conc, axis=0),
        {
            '_FillValue': 255,
            'grid_mapping': 'crs',
        },
        {
            'zlib': True,
        },
    )

    return ecdr_conc_ds


''' This is the code we are working on...
def create_initial_daily_ecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Create netcdf file of ide data set."""
    logger.info(
        'Creating initial daily ECDR netCDF file for:\n'
        f'{date=}, {hemisphere=}, {resolution=}'
    )

    # Initialize xr ds container, including georeferencing coords, dimensions
    # Compute raw bt_conc dataarray
    # Compute raw nt_conc dataarray
    # Compute cdr_conc dataarray
    # Add appropriate global metadata
    # Determine output filename
    # Return output filename
'''


def make_cdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Create the cdr netCDF file."""
    logger.info(f'Creating CDR for {date=}, {hemisphere=}, {resolution=}')
    conc_ds = amsr2_cdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat='u2',
        algorithm='cdr',
        resolution=f'{resolution}km',
    )
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(
        output_path,
        encoding={'conc': {'zlib': True}},
    )
    logger.info(f'Wrote AMSR2 CDR concentration field: {output_path}')


def create_idecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Generate the initial daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_cdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                output_dir=output_dir,
            )
        except Exception:
            logger.error(
                'Failed to create NetCDF for ' f'{hemisphere=}, {date=}, {resolution=}.'
            )
            err_filename = standard_output_filename(
                hemisphere=hemisphere,
                date=date,
                sat='u2',
                algorithm='cdr',
                resolution=f'{resolution}km',
            )
            err_filename += '.error'
            logger.info(f'Writing error info to {err_filename}')
            with open(output_dir / err_filename, 'w') as f:
                traceback.print_exc(file=f)
                traceback.print_exc(file=sys.stdout)


@click.command(name='idecdr')
@click.option(
    '-d',
    '--date',
    required=True,
    type=click.DateTime(formats=('%Y-%m-%d',)),
    callback=datetime_to_date,
)
@click.option(
    '-h',
    '--hemisphere',
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    '-o',
    '--output-dir',
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    '-r',
    '--resolution',
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
) -> None:
    """Run the initial daily ECDR algorithm with AMSR2 data."""
    create_idecdr_for_date_range(
        start_date=date,
        end_date=date,
        gridid=gridid,
        tbsrc=tb_source,
        output_dir=output_dir,
    )


def parse_cmdline_iedcdr_params():
    """Extract info from command line call of initial_daily_ecdr.py."""
    import sys

    print(f'cmdline args: {sys.argv}')
    raise RuntimeError('in parse_cmdline_iedcdr_params')


if __name__ == '__main__':
    # vvvv MODIFY THESE PARAMETERS AS NEEDED vvvv
    start_date, end_date, gridid, tb_source, output_dir = parse_cmdline_iedcdr_params()

    create_idecdr_for_date_range(
        start_date=start_date,
        end_date=end_date,
        gridid=gridid,
        tb_source=tb_source,
        output_dir=output_dir,
    )
    start_date = dt.date(2012, 7, 2)
    end_date = dt.date(2021, 2, 11)
    # resolution: ECDR_ = '12'
    resolution = '12'
    output_dir = CDR_DATA_DIR
    # ^^^^ MODIFY THESE PARAMETERS AS NEEDED ^^^^
    for hemisphere in get_args(Hemisphere):
        create_idecdr_for_date_range(
            start_date=start_date,
            end_date=end_date,
            hemisphere=hemisphere,
            resolution=resolution,
            output_dir=output_dir,
        )