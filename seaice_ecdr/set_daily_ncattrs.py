"""Set the netCDF variables and attributes for ecdr data files."""
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr

from seaice_ecdr.nc_attrs import get_global_attrs

CDECDR_FIELDS_TO_DROP = [
    "h18_day_si",
    "h36_day_si",
    "land_mask",
    "pole_mask",
    "invalid_ice_mask",
    "invalid_tb_mask",
    "bt_weather_mask",
    "nt_weather_mask",
    "conc",
    "cdr_conc_ti",
]

CDECDR_FIELDS_TO_RENAME = {
    "cdr_conc": "cdr_seaice_conc",
    "bt_conc_raw": "raw_bootstrap_seaice_conc",
    "nt_conc_raw": "raw_nasateam_seaice_conc",
    "qa_of_cdr_seaice_conc": "qa_of_cdr_seaice_conc",  # Note: unchanged
    "spatint_bitmask": "spatial_interpolation_flag",  # Note: bitmask, not flag?
    "temporal_flag": "temporal_interpolation_flag",
    # 'stdev_of_seaice_conc': 'stdev_of_cdr_seaice_conc',  # this should be in tiecdr
}


def finalize_cdecdr_ds(
    ds_in: xr.Dataset,
    fields_to_drop: list = CDECDR_FIELDS_TO_DROP,
    fields_to_rename: dict = CDECDR_FIELDS_TO_RENAME,
) -> xr.Dataset:
    """Create the final, published version of the complete daily dataset."""
    # TODO: It may be best to move these attributes and specifications to
    #       a _nc_specifications.py file?
    ds = ds_in.copy(deep=True)

    # Drop unnecessary fields
    # Setting errors to ignore because SH will not have pole hole or melt_onset vars
    ds = ds.drop_vars(fields_to_drop, errors="ignore")

    # Rename fields
    ds = ds.rename_vars(fields_to_rename)

    # Variables that need special handling...

    # cdr_seaice_conc should be ubyte
    ds["cdr_seaice_conc"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc"].data.astype(np.uint8),
        {
            "_FillValue": 255,
            "standard_name": "sea_ice_area_fraction",
            "units": "1",
            "long_name": (
                "NOAA/NSIDC Climate Data Record of Passive Microwave"
                " Sea Ice Concentration"
            ),
            "grid_mapping": "crs",
            # TODO: We may add more flag values later
            "reference": "https://nsidc.org/data/g02202/versions/5/",
            "ancillary_variables": "stdev_of_cdr_seaice_conc qa_of_cdr_seaice_conc",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(0.0),
        },
        {
            "zlib": True,
        },
    )

    # Standard deviation file is converted from 2d to [time, y x] coords
    ds["stdev_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["stdev_of_cdr_seaice_conc"].data, axis=0),
        {
            "_FillValue": -1,
            "long_name": (
                "Passive Microwave Sea Ice"
                "Concentration Source Estimated Standard Deviation"
            ),
            "units": "K",
            "grid_mapping": "crs",
            "valid_range": np.array((0.0, 300.0), dtype=np.float32),
        },
        {
            "zlib": True,
        },
    )

    # TODO: Verify that flag mask values have been set properly
    ds["qa_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["qa_of_cdr_seaice_conc"].data.astype(np.uint8), axis=0),
        {
            "standard_name": "status_flag",
            "long_name": "Passive Microwave Sea Ice Concentration QC flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array(
                (
                    1,  # BT_weather_filter_applied
                    2,  # NT_weather_filter_applied
                    4,  # Land_spillover_filter_applied
                    8,  # No_input_data
                    16,  # Valid_ice_mask_applied
                    32,  # Spatial_interpolation_applied
                    64,  # Temporal_interpolation_applied
                    128,  # Melt_start_detected
                ),
                dtype=np.uint8,
            ),
            "flag_meanings": (
                "BT_weather_filter_applied"  # Note: no leading space
                " NT_weather_filter_applied"
                " Land_spillover_filter_applied"
                " No_input_data"
                " valid_ice_mask_applied"
                " spatial_interpolation_applied"
                " temporal_interpolation_applied"
                " melt_start_detected"
            ),
            "valid_range": np.array((0, 255), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )

    # Note: this is NH only, hence the try/except block
    try:
        ds["melt_onset_day_cdr_seaice_conc"] = (
            ("time", "y", "x"),
            ds["melt_onset_day_cdr_seaice_conc"].data,
            {
                "standard_name": "status_flag",
                "long_name": "Day Of Year of NH Snow Melt Onset On Sea Ice",
                "units": "1",
                "grid_mapping": "crs",
                "valid_range": np.array((60, 255), dtype=np.uint8),
                "comment": (
                    "Value of 255 means no melt detected yet or the date"
                    " is outside the melt season.  Other values indicate"
                    " the day of year when melt was first detected at"
                    " this location."
                ),
            },
            {
                "zlib": True,
            },
        )
    except KeyError:
        # We do not expect melt onset in Southern hemisphere
        pass

    # TODO: Verify that spatial interpolation set properly for pole hole fill
    # TODO: Verify flag mask values and meanings
    ds["spatial_interpolation_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["spatial_interpolation_flag"].data.astype(np.uint8), axis=0),
        {
            "standard_name": "status_flag",
            "long_name": "Passive Microwave Sea Ice Concentration spatial interpolation flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array(
                (
                    0,
                    1,
                    2,
                    4,
                    8,
                    16,
                    32,
                ),
                dtype=np.uint8,
            ),
            "flag_meanings": (
                "no_spatial_interpolation"
                " 19v_tb_value_interpolated"
                " 19h_tb_value_interpolated"
                " 22v_tb_value_interpolated"
                " 37v_tb_value_interpolated"
                " 37h_tb_value_interpolated"
                " Pole_hole_spatially_interpolated_(Arctic_only)"
            ),
            "valid_range": np.array((0, 63), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )

    # Note: cannot have one-sided interpolations of 4- or 5- days, so
    #       the values 4, 5, 40, and 50 are not possible
    ds["temporal_interpolation_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["temporal_interpolation_flag"].data.astype(np.uint8), axis=0),
        {
            "standard_name": "status_flag",
            "long_name": "Passive Microwave Sea Ice Concentration temporal interpolation flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_values": np.array(
                (
                    0,  # no_temporal_interp
                    1,  # 1_day_after
                    2,  # 2_days_after
                    3,  # 3_days_after
                    10,  # 1_day_prior
                    11,  # 1_day_prior_and_1_day_after
                    12,  # 1_day_prior_and_2_days_after
                    13,  # 1_day_prior_and_3_days_after
                    14,  # 1_day_prior_and_4_days_after
                    15,  # 1_day_prior_and_5_days_after
                    20,  # 2_days_prior
                    21,  # 2_days_prior_and_1_day_after
                    22,  # 2_days_prior_and_2_days_after
                    23,  # 2_days_prior_and_3_days_after
                    24,  # 2_days_prior_and_4_days_after
                    25,  # 2_days_prior_and_5_days_after
                    30,  # 3_days_prior
                    31,  # 3_days_prior_and_1_day_after
                    32,  # 3_days_prior_and_2_days_after
                    33,  # 3_days_prior_and_3_days_after
                    34,  # 3_days_prior_and_4_days_after
                    35,  # 3_days_prior_and_5_days_after
                    41,  # 4_days_prior_and_1_day_after
                    42,  # 4_days_prior_and_2_days_after
                    43,  # 4_days_prior_and_3_days_after
                    44,  # 4_days_prior_and_4_days_after
                    45,  # 4_days_prior_and_5_days_after
                    51,  # 5_days_prior_and_1_day_after
                    52,  # 5_days_prior_and_2_days_after
                    53,  # 5_days_prior_and_3_days_after
                    54,  # 5_days_prior_and_4_days_after
                    55,  # 5_days_prior_and_5_days_after
                ),
                dtype=np.uint8,
            ),
            "flag_meanings": (
                "no_temporal_interp "  # These values are copy/pasted from above
                "1_day_after "
                "2_days_after "
                "3_days_after "
                "1_day_prior "
                "1_day_prior_and_1_day_after "
                "1_day_prior_and_2_days_after "
                "1_day_prior_and_3_days_after "
                "1_day_prior_and_4_days_after "
                "1_day_prior_and_5_days_after "
                "2_days_prior "
                "2_days_prior_and_1_day_after "
                "2_days_prior_and_2_days_after "
                "2_days_prior_and_3_days_after "
                "2_days_prior_and_4_days_after "
                "2_days_prior_and_5_days_after "
                "3_days_prior "
                "3_days_prior_and_1_day_after "
                "3_days_prior_and_2_days_after "
                "3_days_prior_and_3_days_after "
                "3_days_prior_and_4_days_after "
                "3_days_prior_and_5_days_after "
                "4_days_prior_and_1_day_after "
                "4_days_prior_and_2_days_after "
                "4_days_prior_and_3_days_after "
                "4_days_prior_and_4_days_after "
                "4_days_prior_and_5_days_after "
                "5_days_prior_and_1_day_after "
                "5_days_prior_and_2_days_after "
                "5_days_prior_and_3_days_after "
                "5_days_prior_and_4_days_after "
                "5_days_prior_and_5_days_after"  # Last has no trailing space!
            ),
            "comment": (
                "Value of 0 indicates no temporal interpolation occurred. "
                " Values greater than 0 and less than 100 are of the form"
                " 'AB' where 'A' indicates the number of days prior to the"
                " current day and 'B' indicates the number of days after the"
                " current day used to linearly interpolate the data.  If"
                " either A or B are zero, the value was extrapolated from"
                " that date rather than interpolated.  A value of 255"
                " indicates that temporal interpolation could not be"
                " accomplished."
            ),
            "valid_range": np.array((0, 55), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )

    # bootstrap: add time dim and convert to ubyte
    # TODO: adding time dimension should probably happen earlier
    # TODO: conversion to ubyte should be done with DataArray encoding dict
    ds["raw_bootstrap_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["raw_bootstrap_seaice_conc"].astype(np.uint8), axis=0),
        # ds["raw_bootstrap_seaice_conc"].attrs,  # We are overwriting these
        # TODO: These should be a standard dictionary of "conc datavar attrs"
        #       ...except the long_name
        {
            "_FillValue": 255,
            "standard_name": "sea_ice_area_fraction",
            "units": "1",
            "long_name": (
                "Bootstrap sea ice concntration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            # TODO: scale_factor and add_offset might get set during encoding
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(0.0),
        },
        {"zlib": True},
    )

    # nasateam: add time dim and convert to ubyte
    # TODO: adding time dimension should probably happen earlier
    # TODO: conversion to ubyte should be done with DataArray encoding dict
    ds["raw_nasateam_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["raw_nasateam_seaice_conc"].astype(np.uint8), axis=0),
        # NOTE: NOT USING ds["raw_nasateam_seaice_conc"].attrs
        {
            "_FillValue": 255,
            "standard_name": "sea_ice_area_fraction",
            "units": "1",
            "long_name": (
                "NASA Team sea ice concntration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            # TODO: scale_factor and add_offset might get set during encoding?
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(0.0),
        },
        {"zlib": True},
    )

    # Finally, address global attributes
    ds_date: dt.date = pd.Timestamp(ds_in.time.values[0]).date()
    new_global_attrs = get_global_attrs(
        time_coverage_start=dt.datetime(
            ds_date.year, ds_date.month, ds_date.day, 0, 0, 0
        ),
        time_coverage_end=dt.datetime(
            ds_date.year, ds_date.month, ds_date.day, 23, 59, 59
        ),
        # TODO: support different resolutions, dataset_ids, platforms, and sensors!
        resolution="12.5",
        source_dataset_id="AU_SI12",
        platform="GCOM-W1 > Global Change Observation Mission 1st-Water",
        sensor="AMSR2 > Advanced Microwave Scanning Radiometer 2",
    )
    ds.attrs.update(new_global_attrs)

    return ds
