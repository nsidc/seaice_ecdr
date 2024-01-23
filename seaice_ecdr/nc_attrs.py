import calendar
import datetime as dt
import subprocess
from collections import OrderedDict
from functools import cache
from typing import Any, Final, Literal, get_args

import pandas as pd
import xarray as xr

from seaice_ecdr._types import SUPPORTED_SAT
from seaice_ecdr.constants import ECDR_PRODUCT_VERSION

# Datetime string format for date-related attributes.
DATE_STR_FMT = "%Y-%m-%dT%H:%M:%SZ"


Temporality = Literal["daily", "monthly"]


@cache
def _get_software_version_id():
    """Return string representing this software's version.

    Takes the form <git_repo>@<git_hash>. E.g.,:
    "git@github.com:nsidc/seaice_ecdr.git@10fdd316452d0d69fbcf4e7915b66c227298b0ec"
    """
    software_git_hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True
    )
    software_git_hash_result.check_returncode()
    software_git_hash = software_git_hash_result.stdout.decode("utf8").strip()

    software_git_url_result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"], capture_output=True
    )
    software_git_url_result.check_returncode()
    software_git_url = software_git_url_result.stdout.decode("utf8").strip()

    software_version_id = f"{software_git_url}@{software_git_hash}"

    return software_version_id


def _get_time_coverage_attrs(
    time: xr.DataArray,
    temporality: Temporality,
    aggregate: bool,
) -> dict[str, Any]:
    """Return a dictionary of time coverage and resolution attrs.

    * `time_coverage_duration`:
        * For the current year’s daily aggregate file, should be PXXD where `XX`
          is the day of year of the last day of data until we get a full year. A
          full year's worth of data gets the value `P1Y`.

          For 1978 aggregate daily file, it will always be P75D since data
          start on Oct 25 1978 (day of year of end date - day of year of the
          first date + 1)

        * For the monthly files, should be PNM where N is the number of months
          (e.g., P1M for a single month, P480M for 480 months in an aggregated
          file).

        * For the 1987-1988 outage, we will plan to call those a full year of
          data.

    * `time_coverage_resolution`:
       * For monthly file this is P1M. For daily, "P1D"
    """
    assert temporality in get_args(Temporality)

    start_date = pd.Timestamp(time.min().values).date()
    start_datetime = dt.datetime(
        start_date.year, start_date.month, start_date.day, 0, 0, 0
    )
    end_date = pd.Timestamp(time.max().values).date()
    if temporality == "monthly":
        # The coverage end should be the last day of the month.
        _, last_day_of_month = calendar.monthrange(end_date.year, end_date.month)
        end_datetime = dt.datetime(
            end_date.year, end_date.month, last_day_of_month, 23, 59, 59
        )
    else:
        end_datetime = dt.datetime(
            end_date.year, end_date.month, end_date.day, 23, 59, 59
        )

    time_coverage_attrs = dict(
        time_coverage_start=start_datetime.strftime(DATE_STR_FMT),
        time_coverage_end=end_datetime.strftime(DATE_STR_FMT),
    )

    if temporality == "daily":
        time_coverage_attrs["time_coverage_duration"] = "P1D"
        time_coverage_attrs["time_coverage_resolution"] = "P1D"

        if aggregate:
            # Check if we have a full year of data
            # TODO: there could be missing days in between Jan 1 and Dec. 31. To
            # be sure we'd have to check the `time` array for every day of the
            # year.
            if (
                start_date.month == 1
                and start_date.day == 1
                and end_date.month == 12
                and end_date.day == 31
            ):
                time_coverage_attrs["time_coverage_duration"] = "P1Y"
            else:
                # For partial years, we use the `PXXD` format where `XX` is
                # the number of days until we get a full year?
                n_days = (end_date - start_date).days + 1
                time_coverage_attrs["time_coverage_duration"] = f"P{n_days}D"
    else:
        time_coverage_attrs["time_coverage_resolution"] = "P1M"

        n_months = len(time)
        if not aggregate:
            assert n_months == 1
        time_coverage_attrs["time_coverage_duration"] = f"P{n_months}M"

    return time_coverage_attrs


# Here’s what the GCMD platform long name should be based on sensor/platform short name:
PLATFORMS_FOR_SATS: dict[SUPPORTED_SAT, str] = dict(
    am2="GCOM-W1 > Global Change Observation Mission 1st-Water",
    ame="Aqua > Earth Observing System, Aqua",
    F17="DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17",
    F13="DMSP 5D-2/F13 > Defense Meteorological Satellite Program-F13",
    F11="DMSP 5D-2/F11 > Defense Meteorological Satellite Program-F11",
    F08="DMSP 5D-2/F8 > Defense Meteorological Satellite Program-F8",
    n07="Nimbus-7",
)


def _unique_sats(sats: list[SUPPORTED_SAT]) -> list[SUPPORTED_SAT]:
    """Return the unique set of satellites.

    Order is preserved.
    """
    # `set` is unordered. This gets the unique list of `sats`.
    unique_sats = list(dict.fromkeys(sats))

    return unique_sats


def get_platforms_for_sats(sats: list[SUPPORTED_SAT]) -> list[str]:
    """Get the unique set of platforms for the given list of sats.

    Assumes `sats` is ordered from oldest->newest.
    """
    # `set` is unordered. This gets the unique list of `sats`.
    unique_sats = _unique_sats(sats)
    platforms_for_sat = [PLATFORMS_FOR_SATS[sat] for sat in unique_sats]

    return platforms_for_sat


# Here’s what the GCMD sensor name should be based on sensor short name:
SENSORS_FOR_SATS: dict[SUPPORTED_SAT, str] = dict(
    am2="AMSR2 > Advanced Microwave Scanning Radiometer 2",
    ame="AMSR-E > Advanced Microwave Scanning Radiometer-EOS",
    F17="SSMIS > Special Sensor Microwave Imager/Sounder",
    # TODO: de-dup SSM/I text?
    F13="SSM/I > Special Sensor Microwave/Imager",
    F11="SSM/I > Special Sensor Microwave/Imager",
    F08="SSM/I > Special Sensor Microwave/Imager",
    n07="SMMR > Scanning Multichannel Microwave Radiometer",
)


def get_sensors_for_sats(sats: list[SUPPORTED_SAT]) -> list[str]:
    """Get the unique set of sensors for the given list of sats.

    Assumes `sats` is ordered from oldest->newest.
    """
    unique_sats = _unique_sats(sats)
    sensors_for_sat = [SENSORS_FOR_SATS[sat] for sat in unique_sats]

    return sensors_for_sat


def get_global_attrs(
    *,
    time: xr.DataArray,
    # daily or monthly?
    temporality: Temporality,
    # Is this an aggregate file, or not?
    aggregate: bool,
    # `source` attribute. Currently passed through unchanged.
    # For daily files, this will be AU_SI12 for AMSR2,
    # AE_SI12 for AMSR-E, and NSIDC-0001 for SSMIS, SSM/I, and SMMR.
    # For monthly and aggregate files, `source` is a comman-space-separated string
    # of source filenames.
    source: str,
    # List of satellites that provided data for the given netcdf file.
    sats: list[SUPPORTED_SAT],
) -> dict[str, Any]:
    """Return a dictionary containing the global attributes for a standard ECDR NetCDF file.

    Note the `date_created` field is populated with the current UTC datetime. It is
    assumed the result of this function will be used to set the attributes of an
    xr.Dataset that will be promptly written to disk as a NetCDF file.
    """

    # TODO: support different resolutions, platforms, and sensors!
    resolution: Final = "12.5"
    platform = ", ".join(get_platforms_for_sats(sats))
    sensor = ", ".join(get_sensors_for_sats(sats))

    time_coverage_attrs = _get_time_coverage_attrs(
        temporality=temporality,
        aggregate=aggregate,
        time=time,
    )

    new_global_attrs = OrderedDict(
        # We expect conventions to be the first item in the global attributes.
        Conventions="CF-1.11, ACDD-1.3",
        # Use the current UTC time to set the `date_created` attribute.
        date_created=dt.datetime.utcnow().strftime(DATE_STR_FMT),
        **time_coverage_attrs,
        title=(
            "NOAA-NSIDC Climate Data Record of Passive Microwave"
            " Sea Ice Concentration Version 5"
        ),
        program="NOAA Climate Data Record Program",
        software_version_id=_get_software_version_id(),
        metadata_link="https://nsidc.org/data/g02202/versions/5/",
        product_version=ECDR_PRODUCT_VERSION,
        spatial_resolution=f"{resolution}km",
        standard_name_vocabulary="CF Standard Name Table (v83, 17 October 2023)",
        id="https://doi.org/10.7265/rjzb-pf78",
        naming_authority="org.doi.dx",
        license="No constraints on data access or use",
        summary="This data set provides a passive microwave sea ice concentration climate data record (CDR) based on gridded brightness temperatures from the Advanced Microwave Scanning Radiometer 2 (AMSR2) onboard the GCOM-W1 satellite, the Advanced Microwave Scanning Radiometer for EOS (AMSR-E) onboard the Aqua satellite, the Special Sensor Microwave Imager (SSM/I) and the Special Sensor Microwave Imager/Sounder (SSMIS) that are part of the Defense Meteorological Satellite Program (DMSP) series of passive microwave radiometers, and the Nimbus-7 Scanning Multichannel Microwave Radiometer (SMMR). The sea ice concentration CDR is an estimate of sea ice concentration that is produced by combining concentration estimates from two algorithms developed at the NASA Goddard Space Flight Center (GSFC): the NASA Team algorithm and the Bootstrap algorithm. The individual algorithms are used to process and combine brightness temperature data at NSIDC. This product is designed to provide a consistent time series of sea ice concentrations (the fraction, or percentage, of ocean area covered by sea ice) from November 1978 to the present which spans the coverage of several passive microwave instruments. The data are gridded on the NSIDC polar stereographic grid with 12.5 x 12.5 km grid cells, and are available in NetCDF file format. Each file contains a variable with the CDR concentration values as well as variables that hold the raw NASA Team and Bootstrap processed concentrations for reference; Variables containing standard deviation, quality flags, and projection information are also included.",
        keywords="EARTH SCIENCE > CRYOSPHERE > SEA ICE > SEA ICE CONCENTRATION, Continent > North America > Canada > Hudson Bay, Geographic Region > Arctic, Geographic Region > Polar, Geographic Region > Northern Hemisphere, Ocean > Arctic Ocean, Ocean > Arctic Ocean > Barents Sea, Ocean > Arctic Ocean > Beaufort Sea, Ocean > Arctic Ocean > Chukchi Sea, CONTINENT > NORTH AMERICA > CANADA > HUDSON BAY, Ocean > Atlantic Ocean > North Atlantic Ocean > Davis Straight, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > GULF OF ST LAWRENCE, Ocean > Atlantic Ocean > North Atlantic Ocean > North Sea, Ocean > Atlantic Ocean > North Atlantic Ocean > Norwegian Sea, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > SVALBARD AND JAN MAYEN, Ocean > Pacific Ocean, Ocean > Pacific Ocean > North Pacific Ocean > Bering Sea, Ocean > Pacific Ocean > North Pacific Ocean > Sea Of Okhotsk",
        keywords_vocabulary="NASA Global Change Master Directory (GCMD) Keywords, Version 17.1",
        cdm_data_type="Grid",
        project="NOAA/NSIDC passive microwave sea ice concentration climate data record",
        creator_url="http://nsidc.org/",
        creator_email="nsidc@nsidc.org",
        institution="NSIDC > National Snow and Ice Data Center",
        processing_level="NOAA Level 3",
        contributor_name="Walter N. Meier, Florence Fetterer, Ann Windnagel, J. Scott Stewart, Trey Stafford",
        contributor_role="principal investigator, author, author, software developer, software developer",
        acknowledgment="This project was supported in part by a grant from the NOAA Climate Data Record Program. The NASA Team and Bootstrap sea ice concentration algorithms were developed by Donald J. Cavalieri, Josefino C. Comiso, Claire L. Parkinson, and others at the NASA Goddard Space Flight Center in Greenbelt, MD.",
        source=source,
        platform=platform,
        sensor=sensor,
        # TODO: ideally, these would get dynamically set from the input data's
        # grid definition...
        geospatial_lat_min=31.35,
        geospatial_lat_max=90.0,
        geospatial_lat_units="degrees_north",
        geospatial_lon_min=-180.0,
        geospatial_lon_max=180.0,
        geospatial_lon_units="degrees_east",
    )

    return new_global_attrs
