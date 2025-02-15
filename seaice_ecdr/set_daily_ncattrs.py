"""Set the netCDF variables and attributes for ecdr data files."""

import numpy as np
import xarray as xr
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    remove_FillValue_from_coordinate_vars,
)
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.tb_data import get_data_url_from_data_source

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

    ds["cdr_seaice_conc"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "NOAA/NSIDC CDR of Passive Microwave" " Sea Ice Concentration"
            ),
            "grid_mapping": "crs",
            "ancillary_variables": "cdr_seaice_conc_stdev cdr_seaice_conc_qa_flag",
            "valid_range": np.array((0, 100), dtype=np.uint8),
        },
        # Note: encoding is set when saved to netcdf file
    )

    # Standard deviation file is converted from 2d to [time, y x] coords
    ds["cdr_seaice_conc_stdev"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc_stdev"].data,
        {
            "_FillValue": -1,
            "long_name": (
                "NOAA/NSIDC CDR of Passive Microwave Sea Ice"
                " Concentration Source Estimated Standard Deviation"
            ),
            "units": "1",
            "grid_mapping": "crs",
            "valid_range": np.array((0.0, 1.0), dtype=np.float32),
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

    ds["cdr_seaice_conc_qa_flag"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc_qa_flag"].data.astype(np.uint8),
        {
            "standard_name": "status_flag",
            "long_name": "NOAA/NSIDC CDR of Passive Microwave Sea Ice Concentration QA flags",
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
    # Note: valid range allows values:
    #        0: conc < 50% at start of melt season
    #   60-244: day-of-year melt detected during melt season
    #      255: no melt detected during melt season
    try:
        ds["cdr_melt_onset_day"] = (
            ("time", "y", "x"),
            ds["cdr_melt_onset_day"].data,
            {
                "standard_name": "status_flag",
                "long_name": "NOAA/NSIDC CDR Day Of Year of NH Snow Melt Onset On Sea Ice",
                "units": "1",
                "grid_mapping": "crs",
                "valid_range": np.array((0, 255), dtype=np.uint8),
                "comment": (
                    "Value of 0 indicates sea ice concentration less than 50%"
                    " at start of melt season; values of 60-244 indicate day"
                    " of year of snow melt onset on sea ice detected during"
                    " melt season; value of 255 indicates no melt detected"
                    " during melt season, including non-ocean grid cells."
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

    ds["cdr_seaice_conc_interp_spatial_flag"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc_interp_spatial_flag"].data.astype(np.uint8),
        {
            "standard_name": "status_flag",
            "long_name": "NOAA/NSIDC CDR of Passive Microwave Sea Ice Concentration spatial interpolation flags",
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
    ds["cdr_seaice_conc_interp_temporal_flag"] = (
        ("time", "y", "x"),
        ds["cdr_seaice_conc_interp_temporal_flag"].data.astype(np.uint8),
        {
            "standard_name": "status_flag",
            "long_name": "NOAA/NSIDC CDR of Passive Microwave Sea Ice Concentration temporal interpolation flags",
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
                " Values greater than 0 and less than or equal to 55 are of the form"
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
    # NOTE: We allow raw siconc up to 254% because (1) that is the maximum
    #       representable value for a non-negative conc with a _FlagValue
    #       for missing of 255, and (2) for potential validation measures.
    ds["raw_bt_seaice_conc"] = (
        ("time", "y", "x"),
        ds["raw_bt_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "NOAA/NSIDC CDR of Bootstrap sea ice concentration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 254), dtype=np.uint8),
        },
    )

    # NOTE: We are overwriting the attrs of the original conc field
    # TODO: adding time dimension should probably happen earlier
    # TODO: conversion to ubyte should be done with DataArray encoding dict
    # TODO: scale_factor and add_offset might get set during encoding
    # NOTE: We allow raw siconc up to 254% because (1) that is the maximum
    #       representable value for a non-negative conc with a _FlagValue
    #       for missing of 255, and (2) for potential validation measures.
    ds["raw_nt_seaice_conc"] = (
        ("time", "y", "x"),
        ds["raw_nt_seaice_conc"].data,
        {
            "standard_name": "sea_ice_area_fraction",
            "coverage_content_type": "image",
            "units": "1",
            "long_name": (
                "NOAA/NSIDC CDR of NASA Team sea ice concentration;"
                " raw field with no masking or filtering"
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 254), dtype=np.uint8),
        },
    )

    # Finally, address global attributes
    new_global_attrs = get_global_attrs(
        time=ds.time,
        temporality="daily",
        aggregate=False,
        source=f"Generated from {get_data_url_from_data_source(data_source=ds_in.data_source)}",
        platform_ids=[ds_in.platform],
        resolution=resolution,
        hemisphere=hemisphere,
        ancillary_source=ancillary_source,
    )
    ds.attrs = new_global_attrs

    # Coordinate values should not have _FillValue set
    ds = remove_FillValue_from_coordinate_vars(ds)

    # Note: Here, the x and y coordinate variables *do* have valid_range set

    return ds
