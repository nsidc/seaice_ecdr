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
from pm_icecon.fill_polehole import fill_pole_hole
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.land_spillover import (
    apply_nt2a_land_spillover,
    apply_nt2b_land_spillover,
    load_or_create_land90_conc,
    read_adj123_file,
)
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_icecon.util import date_range, standard_output_filename

from seaice_ecdr.gridid_to_xr_dataarray import get_dataset_for_gridid


def xwm(m='exiting in xwm()'):
    raise SystemExit(m)


def cdr_bootstrap(
    date: dt.date,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_coefs,
):
    """Generate the raw bootstrap concentration field."""
    wtp_37v = bt_coefs['bt_wtp_v37']
    wtp_37h = bt_coefs['bt_wtp_h37']
    wtp_19v = bt_coefs['bt_wtp_v19']

    itp_37v = bt_coefs['bt_itp_v37']
    itp_37h = bt_coefs['bt_itp_h37']
    itp_19v = bt_coefs['bt_itp_v19']

    bt_conc = bt.calc_bootstrap_conc(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        wtp_37v=wtp_37v,
        wtp_37h=wtp_37h,
        wtp_19v=wtp_19v,
        itp_37v=itp_37v,
        itp_37h=itp_37h,
        itp_19v=itp_19v,
        line_37v37h=bt_coefs['vh37_lnline'],
        line_37v19v=bt_coefs['v1937_lnline'],
        ad_line_offset=bt_coefs['ad_line_offset'],
        maxic_frac=bt_coefs['maxic'],
        missing_flag_value=DEFAULT_FLAG_VALUES.missing,
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
        assert tb_v37.shape == bt_tb_mask.shape
    except AssertionError as e:
        raise ValueError(f'Mismatched shape error in get_bt_tb_mask\n{e}')

    return bt_tb_mask


def calculate_cdr_conc_raw(
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    missing_flag_value,
    bt_coefs,
    nt_coefs,
    #cdr_fields,
    #bt_weather_mask: npt.NDArray,
    #nt_weather_mask: npt.NDArray,
    #invalid_ice_mask: npt.NDArray,
) -> npt.NDArray:
    """Run the CDR algorithm."""
    # First, get bootstrap conc.
    bt_conc = cdr_bootstrap(
        date,
        tb_v37,
        tb_h37,
        tb_v19,
        tb_v22,
        bt_coefs,
    )

    # Next, get nasateam conc. Note that concentrations from nasateam may be
    # >100%.
    nt_conc = cdr_nasateam(
        date,
        tb_h19,
        tb_v37,
        tb_v19,
        nt_coefs['nt_tiepoints'],
    )

    # Now calculate CDR SIC
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice
    # Note: Here, values without sea ice (because no TBs) have val np.nan
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    return cdr_conc


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
    ecdr_ide_ds = get_dataset_for_gridid(gridid, date)

    # Set initial global attributes
    ecdr_ide_ds.attrs['description'] = 'Initial daily cdr conc file'

    file_date = \
        dt.date(1970, 1, 1) \
        + dt.timedelta(days=int(ecdr_ide_ds.variables["time"].data))
    ecdr_ide_ds.attrs['time_coverage_start'] = \
        str(dt.datetime(file_date.year, file_date.month, file_date.day, 0, 0, 0))
    ecdr_ide_ds.attrs['time_coverage_end'] = \
        str(dt.datetime(file_date.year, file_date.month, file_date.day, 23, 59, 59))

    # Get AU_SI TBs
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # Move TBs to ecdr_ds
    for tbname in ('h18', 'v18', 'v23', 'h36', 'v36'):
        tb_varname = f'{tbname}_day'
        tbdata = xr_tbs.variables[tbname].data
        freq = tbname[1:]
        pol = tbname[:1]
        tb_longname = f'Daily TB {freq}{pol} from AU_SI{resolution}'
        tb_units = 'K'
        ecdr_ide_ds[tb_varname] = (
            ('y', 'x'),
            tbdata,
            {
                '_FillValue': 0,
                'grid_mapping': 'crs',
                'standard_name': 'brightness_temperature',
                'long_name': tb_longname,
                'units': tb_units,
                'valid_range': [np.float64(10.0), np.float64(350.0)],
            },
            {
                'zlib': True,
            },
        )

    # Spatially interpolate the brightness temperatures
    for tbname in ('h18', 'v18', 'v23', 'h36', 'v36'):
        tb_day_name = f'{tbname}_day'
        tb_si_varname = f'{tb_day_name}_si'
        tb_si_data = spatial_interp_tbs(ecdr_ide_ds[tb_day_name].data)
        freq = tbname[1:]
        pol = tbname[:1]
        tb_si_longname = f'Spatially interpolated {ecdr_ide_ds[tb_day_name].long_name}'
        tb_units = 'K'
        ecdr_ide_ds[tb_si_varname] = (
            ('y', 'x'),
            tb_si_data,
            {
                '_FillValue': 0,
                'grid_mapping': 'crs',
                'standard_name': 'brightness_temperature',
                'long_name': tb_si_longname,
                'units': tb_units,
                'valid_range': [np.float64(10.0), np.float64(350.0)],
            },
            {
                'zlib': True,
            },
        )

    bt_coefs_init = pmi_bt_params.get_bootstrap_params(
        date=date,
        satellite='amsr2',
        gridid=gridid,
    )

    # Add the function that generates the bt_tb_mask to bt_coefs
    bt_coefs_init['bt_tb_data_mask_function'] = bt.tb_data_mask

    # TODO: changet this to bt_fields, then assign to ecdr_ide_ds
    bt_fields = pmi_bt_params.get_bootstrap_fields(
        date=date,
        satellite='amsr2',
        gridid=gridid,
    )
    cdr_fields = {key: value for key, value in bt_fields.items()}

    print(f'cdr_fields keys: {cdr_fields.keys()}')
    # pmi_bt_params.get_bootstrap_fields yields:
    #  invalid_ice_mask
    #  land_mask
    #  pole_mask

    # Encode invalid_ice_mask
    ecdr_ide_ds['invalid_ice_mask'] = (
        ('y', 'x'),
        bt_fields['invalid_ice_mask'],
        {
            '_FillValue': 0,
            'grid_mapping': 'crs',
            'standard_name': 'seaice_binary_mask',
            'long_name': 'invalid ice mask',
            'comment': 'Mask indicating where seaice will not exist on this day based on climatology',
            'units': 1,
            'valid_range': [0, 1],
        },
        {
            'zlib': True,
        },
    )

    # Encode land_mask
    ecdr_ide_ds['land_mask'] = (
        ('y', 'x'),
        bt_fields['land_mask'],
        {
            '_FillValue': 0,
            'grid_mapping': 'crs',
            'standard_name': 'land_binary_mask',
            'long_name': 'land mask',
            'comment': 'Mask indicating where land is',
            'units': 1,
        },
        {
            'zlib': True,
        },
    )

    # Determine the NT fields and coefficients
    nt_params = nt_amsr2_params.get_amsr2_params(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    nt_coefs = {}
    nt_coefs['nt_tiepoints'] = nt_params.tiepoints
    nt_coefs['nt_gradient_thresholds'] = nt_params.gradient_thresholds

    cdr_fields['shoremap'] = nt_params.shoremap
    cdr_fields['minic'] = nt_params.minic

    cdr_fields['bt_tb_mask'] = get_bt_tb_mask(
        tb_v37=ecdr_ide_ds['v36_day_si'].data,
        tb_h37=ecdr_ide_ds['h36_day_si'].data,
        tb_v19=ecdr_ide_ds['v18_day_si'].data,
        tb_v22=ecdr_ide_ds['v23_day_si'].data,

        mintb=bt_coefs_init['mintb'],
        maxtb=bt_coefs_init['maxtb'],

        tb_data_mask_function=bt_coefs_init['bt_tb_data_mask_function'],
    )

    # Compute the weather mask
    cdr_fields['bt_weather_mask'] = bt.get_weather_mask_v2(
        v37=ecdr_ide_ds['v36_day_si'].data,
        h37=ecdr_ide_ds['h36_day_si'].data,
        v22=ecdr_ide_ds['v23_day_si'].data,
        v19=ecdr_ide_ds['v18_day_si'].data,

        #land_mask=cdr_fields['land_mask'],
        land_mask=ecdr_ide_ds['land_mask'].data,
        tb_mask=cdr_fields['bt_tb_mask'],
        ln1=bt_coefs_init['vh37_lnline'],
        date=date,
        wintrc=bt_coefs_init['wintrc'],
        wslope=bt_coefs_init['wslope'],
        wxlimt=bt_coefs_init['wxlimt'],
    )

    # Update the Bootstrap coefficients...
    bt_coefs_to_update = (
        'line_37v37h',
        'bt_wtp_v37',
        'bt_wtp_h37',
        'bt_wtp_v19',
        'ad_line_offset',
        'line_37v19v',
    )

    bt_coefs = bt_coefs_init.copy()
    for coef in bt_coefs_to_update:
        bt_coefs.pop(coef, None)

    bt_coefs['vh37_lnline'] = bt.get_linfit(
        land_mask=ecdr_ide_ds['land_mask'].data,
        tb_mask=cdr_fields['bt_tb_mask'],
        tbx=ecdr_ide_ds['v36_day_si'].data,
        tby=ecdr_ide_ds['h36_day_si'].data,
        lnline=bt_coefs_init['vh37_lnline'],
        add=bt_coefs['add1'],
        weather_mask=cdr_fields['bt_weather_mask'],
    )

    bt_coefs['bt_wtp_v37'] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init['bt_wtp_v37'],
        weather_mask=cdr_fields['bt_weather_mask'],
        tb=ecdr_ide_ds['v36_day_si'].data,
    )

    bt_coefs['bt_wtp_h37'] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init['bt_wtp_h37'],
        weather_mask=cdr_fields['bt_weather_mask'],
        tb=ecdr_ide_ds['h36_day_si'].data,
    )

    bt_coefs['bt_wtp_v19'] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init['bt_wtp_v19'],
        weather_mask=cdr_fields['bt_weather_mask'],
        tb=ecdr_ide_ds['v18_day_si'].data,
    )

    bt_coefs['ad_line_offset'] = bt.get_adj_ad_line_offset_v2(
        wtp_x=bt_coefs['bt_wtp_v37'],
        wtp_y=bt_coefs['bt_wtp_h37'],
        line_37v37h=bt_coefs['vh37_lnline']
    )

    bt_coefs['v1937_lnline'] = bt.get_linfit(
        land_mask=ecdr_ide_ds['land_mask'].data,
        tb_mask=cdr_fields['bt_tb_mask'],

        tbx=ecdr_ide_ds['v36_day_si'].data,
        tby=ecdr_ide_ds['v18_day_si'].data,

        lnline=bt_coefs_init['v1937_lnline'],
        add=bt_coefs['add2'],
        weather_mask=cdr_fields['bt_weather_mask'],

        tba=ecdr_ide_ds['h36_day_si'].data,
        iceline=bt_coefs['vh37_lnline'],
        ad_line_offset=bt_coefs['ad_line_offset'],
    )

    # finally, compute the CDR.
    # HERE, probably need cdr_attrs like:
    #   gridid
    #   missing_flag_value
    #   date
    # Also, snould the ields be xarray DataArrays

    cdr_conc_raw = calculate_cdr_conc_raw(
        date=date,
        tb_h19=ecdr_ide_ds['h18_day_si'].data,
        tb_v37=ecdr_ide_ds['v36_day_si'].data,
        tb_h37=ecdr_ide_ds['h36_day_si'].data,
        tb_v19=ecdr_ide_ds['v18_day_si'].data,
        tb_v22=ecdr_ide_ds['v23_day_si'].data,
        missing_flag_value=DEFAULT_FLAG_VALUES.missing,
        bt_coefs=bt_coefs,
        nt_coefs=nt_coefs,
        #cdr_fields=cdr_fields,
        #invalid_ice_mask=ecdr_ide_ds['invalid_ice_mask'].data,
    )

    # Apply masks
    # Get Nasateam weather filter
    nt_gr_2219 = nt.compute_ratio(
        #tb_v22,
        #tb_v19,
        ecdr_ide_ds['v23_day_si'].data,
        ecdr_ide_ds['v18_day_si'].data,
    )
    nt_gr_3719 = nt.compute_ratio(
        #tb_v37,
        #tb_v19,
        ecdr_ide_ds['v36_day_si'].data,
        ecdr_ide_ds['v18_day_si'].data,
    )
    cdr_fields['nt_weather_mask'] = nt.get_weather_filter_mask(
        gr_2219=nt_gr_2219,
        gr_3719=nt_gr_3719,
        gr_2219_threshold=nt_coefs['nt_gradient_thresholds']['2219'],
        gr_3719_threshold=nt_coefs['nt_gradient_thresholds']['3719'],
    )

    set_to_zero_sic = (
        cdr_fields['nt_weather_mask']
        | cdr_fields['bt_weather_mask']
        | cdr_fields['bt_weather_mask']
        | ecdr_ide_ds['invalid_ice_mask'].data
    )

    cdr_conc = cdr_conc_raw.copy()
    cdr_conc[set_to_zero_sic] = 0

    # Apply land spillover corrections
    # TODO: eventually, we want each of these routines to return a e.g.,
    #   delta that can be applied to the input concentration
    #   instead of returning a new conc. Then we would have a
    #   seprate algorithm for choosing how to apply
    #   multiple spillover deltas to a given conc field.

    use_only_nt2_spillover = True

    tb_h19 = ecdr_ide_ds['h18_day_si'].data
    if use_only_nt2_spillover:
        logger.info('Applying NT2 land spillover technique...')
        if tb_h19.shape == (896, 608):
            # NH
            l90c = load_or_create_land90_conc(
                gridid='psn12.5',
                xdim=608,
                ydim=896,
                overwrite=False,
            )
            adj123 = read_adj123_file(
                gridid='psn12.5',
                xdim=608,
                ydim=896,
            )
            cdr_conc = apply_nt2a_land_spillover(cdr_conc, adj123)
            cdr_conc = apply_nt2b_land_spillover(cdr_conc, adj123, l90c)
        elif tb_h19.shape == (664, 632):
            # SH
            l90c = load_or_create_land90_conc(
                gridid='pss12.5',
                xdim=632,
                ydim=664,
                overwrite=False,
            )
            adj123 = read_adj123_file(
                gridid='pss12.5',
                xdim=632,
                ydim=664,
            )
            cdr_conc = apply_nt2a_land_spillover(cdr_conc, adj123)
            cdr_conc = apply_nt2b_land_spillover(cdr_conc, adj123, l90c)

        else:
            raise SystemExit(
                'Could not determine hemisphere from tb shape: {tb_h19.shape}'
            )
    else:
        # nasateam first:
        logger.info('Applying NASA TEAM land spillover technique...')
        cdr_conc = nt.apply_nt_spillover(
            conc=cdr_conc,
            shoremap=cdr_fields['shoremap'],
            minic=cdr_fields['minic'],
        )
        # then bootstrap:
        logger.info('Applying Bootstrap land spillover technique...')
        cdr_conc = bt.coastal_fix(
            conc=cdr_conc,
            missing_flag_value=missing_flag_value,
            land_mask=ecdr_ide_ds['land_mask'].data,
            minic=bt_coefs['minic'],
        )
    # Fill the NH pole hole
    if cdr_conc.shape == (896, 608):
        cdr_conc = fill_pole_hole(cdr_conc)

    # Apply land flag value and clamp max conc to 100.
    # TODO: extract this func from nt and allow override of flag values
    cdr_conc = nt._clamp_conc_and_set_flags(
        shoremap=cdr_fields['shoremap'],
        conc=cdr_conc,
    )

    ecdr_ide_ds['conc'] = (
        ('time', 'y', 'x'),
        np.expand_dims(cdr_conc, axis=0),
        {
            '_FillValue': 255,
            'grid_mapping': 'crs',
            'standard_name': 'sea_ice_area_fraction',
            'long_name': 'Sea ice concentration',
        },
        {
            'zlib': True,
        },
    )

    return ecdr_ide_ds


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
