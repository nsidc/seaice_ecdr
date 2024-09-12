"""Set the netCDF variables and attributes for ecdr data files."""

import numpy as np
import xarray as xr
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.util import get_num_missing_pixels

CDECDR_FIELDS_TO_DROP = [
    "h19_day_si",
    "h37_day_si",
    "non_ocean_mask",
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
}


def finalize_cdecdr_ds(
    ds_in: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
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

    num_missing_conc_pixels = get_num_missing_pixels(
        seaice_conc_var=ds["cdr_seaice_conc"],
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    ds["cdr_seaice_conc"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "NOAA/NSIDC Climate Data Record of Passive Microwave"
                " Sea Ice Concentration"
            ),
            "grid_mapping": "crs",
            "reference": "https://nsidc.org/data/g02202/versions/5",
            "ancillary_variables": "stdev_of_cdr_seaice_conc cdr_qa_seaice_conc",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            "number_of_missing_pixels": num_missing_conc_pixels,
        },
        # Note: encoding is set when saved to netcdf file
    )

    # Standard deviation file is converted from 2d to [time, y x] coords
    ds["stdev_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        ds["stdev_of_cdr_seaice_conc"].data,
        {
            "_FillValue": -1,
            "long_name": (
                "Passive Microwave Sea Ice"
                "Concentration Source Estimated Standard Deviation"
            ),
            "units": "1",
            "grid_mapping": "crs",
            "valid_range": np.array((0.0, 300.0), dtype=np.float32),
        },
        {
            "zlib": True,
        },
    )

    # TODO: Verify that flag mask values have been set properly
    # TODO: Use common dict with key/vals for flag masks/meanings
    qa_flag_masks = [
        1,  # BT_weather_filter_applied
        2,  # NT_weather_filter_applied
        4,  # Land_spillover_filter_applied
        8,  # No_input_data
        16,  # invalid_ice_mask_applied
        32,  # Spatial_interpolation_applied
        64,  # Temporal_interpolation_applied
    ]
    qa_flag_meanings = (
        "BT_weather_filter_applied"  # Note: no leading space
        " NT_weather_filter_applied"
        " Land_spillover_filter_applied"
        " No_input_data"
        " invalid_ice_mask_applied"
        " spatial_interpolation_applied"
        " temporal_interpolation_applied"
    )

    # Melt only occurs in the northern hemisphere. Don't add status flags for SH
    # here.
    if hemisphere == NORTH:
        qa_flag_masks.append(128)  # Melt_start_detected
        qa_flag_meanings += " melt_start_detected"

    ds["cdr_qa_seaice_conc"] = (
        ("time", "y", "x"),
        ds["cdr_qa_seaice_conc"].data.astype(np.uint8),
        {
            "standard_name": "status_flag",
            "long_name": "Passive Microwave Sea Ice Concentration QC flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array(qa_flag_masks, dtype=np.uint8),
            "flag_meanings": qa_flag_meanings,
            "valid_range": np.array((0, sum(qa_flag_masks)), dtype=np.uint8),
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

    # TODO: Use common dict with key/vals for flag masks/meanings
    spatial_interp_flag_masks = [
        1,
        2,
        4,
        8,
        16,
    ]
    spatial_interp_flag_meanings = (
        "19v_tb_value_interpolated"
        " 19h_tb_value_interpolated"
        " 22v_tb_value_interpolated"
        " 37v_tb_value_interpolated"
        " 37h_tb_value_interpolated"
    )
    if hemisphere == NORTH:
        spatial_interp_flag_masks.append(32)
        spatial_interp_flag_meanings += (
            " pole_hole_spatially_interpolated_(Arctic_only)"
        )

    ds["cdr_spatial_interpolation_flag"] = (
        ("time", "y", "x"),
        ds["cdr_spatial_interpolation_flag"].data.astype(np.uint8),
        {
            "standard_name": "status_flag",
            "long_name": "Passive Microwave Sea Ice Concentration spatial interpolation flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array(spatial_interp_flag_masks, dtype=np.uint8),
            "flag_meanings": spatial_interp_flag_meanings,
            "valid_range": np.array(
                (0, sum(spatial_interp_flag_masks)), dtype=np.uint8
            ),
        },
        {
            "zlib": True,
        },
    )

    # Note: cannot have one-sided interpolations of 4- or 5- days, so
    #       the values 4, 5, 40, and 50 are not possible
    # TODO: Use common dict with key/vals for flag masks/meanings
    ds["cdr_temporal_interpolation_flag"] = (
        ("time", "y", "x"),
        ds["cdr_temporal_interpolation_flag"].data.astype(np.uint8),
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

    # TODO: conversion to ubyte should be done with DataArray encoding dict
    # NOTE: We are overwriting the attrs of the original conc field
    # TODO: scale_factor and add_offset might get set during encoding
    ds["raw_bt_seaice_conc"] = (
        ("time", "y", "x"),
        ds["raw_bt_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "Bootstrap sea ice concntration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 100), dtype=np.uint8),
        },
    )

    # NOTE: We are overwriting the attrs of the original conc field
    # TODO: adding time dimension should probably happen earlier
    # TODO: conversion to ubyte should be done with DataArray encoding dict
    # TODO: scale_factor and add_offset might get set during encoding
    ds["raw_nt_seaice_conc"] = (
        ("time", "y", "x"),
        ds["raw_nt_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "NASA Team sea ice concntration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            # We set a `valid_min` of 0 because we allow nasateam raw
            # concentrations >100%. We do not set an upper limit.
            "valid_min": 0,
        },
    )

    # Finally, address global attributes
    new_global_attrs = get_global_attrs(
        time=ds.time,
        temporality="daily",
        aggregate=False,
        source=f"Generated from {ds_in.data_source}",
        platform_ids=[ds_in.platform],
    )
    ds.attrs = new_global_attrs

    # Coordinate values should not have _FillValue set
    ds.time.encoding["_FillValue"] = None
    ds.x.encoding["_FillValue"] = None
    ds.y.encoding["_FillValue"] = None

    return ds
