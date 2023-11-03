"""Set the netCDF variables and attributes for ecdr data files."""

import numpy as np
import xarray as xr
from loguru import logger

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
    "bt_conc_raw": "bootstrap_seaice_conc_raw",
    "nt_conc_raw": "nasateam_seaice_conc_raw",
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
                "Sea Ice Concentration"
            ),
            "grid_mapping": "crs",
            "flag_values": np.array((241, 252, 253, 254, 255), dtype=np.uint8),
            "flag_meanings": "pole_hole lakes coastal land_mask missing_data",
            "reference": "https://nsidc.org/data/g02202/versions/5/",
            "ancillary_variables": "stdev_of_cdr_seaice_conc qa_of_cdr_seaice_conc",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(0.0),
            # TODO: add packing description
        },
        {
            "zlib": True,
        },
    )

    # **************************************
    # This needs to get created in tiecdr...
    # **************************************
    logger.warning("Creating dummy stdev_of_cdr_seaice_conc variable")
    ds["stdev_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        np.zeros(ds["cdr_seaice_conc"].shape, dtype=np.float32),
        {
            "_FillValue": -1,
            "long_name": "Passive Microwave Daily Northern Hemisphere Sea Ice Concentration Source Estimated Standard Deviation",
            "units": "1",
            "grid_mapping": "crs",
            "flag_values": np.array((241, 252, 253, 254, 255), dtype=np.uint8),
            "flag_meanings": "pole_hole lakes coastal land_mask missing_data",
            "valid_range": np.array((0, 100), dtype=np.uint8),
            "scale_factor": np.float32(0.01),
            "add_offset": np.float32(0.0),
            # TODO: add packing description
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
            "long_name": "Passive Microwave Daily Northern Hemisphere Sea Ice Concentration QC flags",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array((1, 2, 4, 8, 16, 32, 64, 128), dtype=np.uint8),
            "flag_meanings": "BT_weather_filter_applied NT_weather_filter_applied BT_land_spillover_filter_applied NT_land_spillover_filter_applied valid_ice_mask_applied spatial_interpolation_applied temporal_interpolation_applied melt_start_detected",
            "valid_range": np.array((0, 255), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )
    # melt onset is set here simply to update attributes
    # TODO: this is NH only

    try:
        ds["melt_onset_day_cdr_seaice_conc"] = (
            ("time", "y", "x"),
            ds["melt_onset_day_cdr_seaice_conc"].data,
            {
                "standard_name": "status_flag",
                "long_name": "Day of NH Snow Melt Onset On Sea Ice",
                "units": "1",
                "grid_mapping": "crs",
                "valid_range": np.array((60, 244), dtype=np.uint8),
                "comment": (
                    "Value of 255 means no melt detected yet or the date is outside"
                    "the melt season.  Other values indicate the day of year when"
                    "melt was first detected at this location."
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
            "long_name": "spatial_interpolation_flag",
            "units": "1",
            "grid_mapping": "crs",
            "flag_masks": np.array((1, 2, 4, 8, 16, 32, 64, 128), dtype=np.uint8),
            "flag_meanings": "19v_tb_value_interpolated 19h_tb_value_interpolated 22v_tb_value_interpolated 37v_tb_value_interpolated 37h_tb_value_interpolated Pole_hole_spatially_interpolated_(Arctic_only)",
            "valid_range": np.array((0, 255), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )

    # TODO: Verify flag mask values and meanings
    # TODO: Verify flag mask meaning strings
    ds["temporal_interpolation_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["temporal_interpolation_flag"].data.astype(np.uint8), axis=0),
        {
            "standard_name": "status_flag",
            "long_name": "temporal_interpolation_flag",
            "units": "1",
            "grid_mapping": "crs",
            "flag_values": np.array(
                (
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    100,
                ),
                dtype=np.uint8,
            ),
            "flag_meanings": "1_day_following 2_days_following 3_days_following 4_days_following 5_days_following 1_day_prior mean_of_prior_and_following_1_day 2_days_prior mean_of_prior_and_following_2_days 3_days_prior mean_of_prior_and_following_3_days 4_days_prior mean_of_prior_and_following_4_days 5_days_prior mean_of_prior_and_following_5_days",
            "comment": 'Value of 0 indicates no temporal interpolation occurred.  Values greater than 0 and less than 100 are of the form "AB" where "A" indicates the number of days prior to the current day and "B" indicates the number of days after the current day used to linearly interpolate the data.  If either A or B are zero, the value was extrapolated from that date rather than interpolated.  A value of 255 indicates that temporal interpolation could not be accomplished.',
            "valid_range": np.array((0, 55), dtype=np.uint8),
        },
        {
            "zlib": True,
        },
    )

    # bootstrap: add time dim and convert to ubyte
    ds["bootstrap_seaice_conc_raw"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["bootstrap_seaice_conc_raw"].astype(np.uint8), axis=0),
        ds["bootstrap_seaice_conc_raw"].attrs,
        {"zlib": True},
    )

    # nasateam: add time dim and convert to ubyte
    ds["nasateam_seaice_conc_raw"] = (
        ("time", "y", "x"),
        np.expand_dims(ds["nasateam_seaice_conc_raw"].astype(np.uint8), axis=0),
        ds["bootstrap_seaice_conc_raw"].attrs,
        {"zlib": True},
    )

    # Finally, address global attributes
    new_global_attrs = {
        "title": (
            "NOAA-NSIDC Climate Data Record of Passive Microwave"
            "Sea Ice Concentration Version 5"
        ),
        "Conventions": "CF-1.10, ACDD-1.3",
        "program": "NOAA Climate Data Record Program",
        "cdr_variable": "cdr_seaice_conc",
        "software_version_id": "tbd",
        "metadata_link": "tbd",
        "product_version": "v05r00",
        "spatial_resolution": "25km",
        "standard_name_vocabulary": "CF Standard Name Table (v16, 11 October 2010)",
        "id": "tbd",
        "naming_authority": "org.doi.dx",
        "license": "No constraints on data access or use",
        "summary": (
            "<tbd> This data set provides a passive microwave sea ice concentration"
            "climate data record (CDR)...."
        ),
        "keywords": "EARTH SCIENCE > CRYOSPHERE > SEA ICE > SEA ICE CONCENTRATION, Continent > North America > Canada > Hudson Bay, Geographic Region > Arctic, Geographic Region > Polar, Geographic Region > Northern Hemisphere, Ocean > Arctic Ocean, Ocean > Arctic Ocean > Barents Sea, Ocean > Arctic Ocean > Beaufort Sea, Ocean > Arctic Ocean > Chukchi Sea, CONTINENT > NORTH AMERICA > CANADA > HUDSON BAY, Ocean > Atlantic Ocean > North Atlantic Ocean > Davis Straight, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > GULF OF ST LAWRENCE, Ocean > Atlantic Ocean > North Atlantic Ocean > North Sea, Ocean > Atlantic Ocean > North Atlantic Ocean > Norwegian Sea, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > SVALBARD AND JAN MAYEN, Ocean > Pacific Ocean, Ocean > Pacific Ocean > North Pacific Ocean > Bering Sea, Ocean > Pacific Ocean > North Pacific Ocean > Sea Of Okhotsk",
        "keywords_vocabulary": "NASA Global Change Master Directory (GCMD) Keywords, Version 7.0.0",
        "cdm_data_type": "Grid",
        "project": "NOAA/NSIDC passive microwave sea ice concentration climate data record",
        "creator_url": "http://nsidc.org/",
        "creator_email": "nsidc@nsidc.org",
        "institution": "NSIDC > National Snow and Ice Data Center",
        "processing_level": "NOAA Level 3",
        "contributor_name": "tbd",
        "contributor_role": "tbd",
        "acknowledgment": "tbd",
        "source": "tbd",
        "platform": "tbd",
        "sensor": "tbd",
    }
    ds.attrs.update(new_global_attrs)

    return ds
