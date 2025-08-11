import calendar
import datetime as dt
import subprocess
from collections import OrderedDict
from functools import cache
from pathlib import Path
from typing import Any, Literal, get_args

import pandas as pd
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    get_ancillary_ds,
)
from seaice_ecdr.constants import ECDR_PRODUCT_VERSION
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID

# Datetime string format for date-related attributes.
DATE_STR_FMT = "%Y-%m-%dT%H:%M:%SZ"


Temporality = Literal["daily", "monthly"]


@cache
def _get_software_version_id():
    """Return string representing this software's version.

    Takes the form <git_repo>@<git_hash>. E.g.,:
    "git@github.com:nsidc/seaice_ecdr.git@10fdd316452d0d69fbcf4e7915b66c227298b0ec"
    """
    _this_dir = Path(__file__).parent
    software_git_hash_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        cwd=_this_dir,
    )
    software_git_hash_result.check_returncode()
    software_git_hash = software_git_hash_result.stdout.decode("utf8").strip()

    software_git_url_result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        cwd=_this_dir,
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


def get_global_attrs(
    *,
    time: xr.DataArray,
    # daily or monthly?
    temporality: Temporality,
    # Is this an aggregate file, or not?
    aggregate: bool,
    # `source` attribute. Currently passed through unchanged.
    # For daily files, this will be NSIDC-0802 for AMSR2,
    # AE_SI12 for AMSR-E, and NSIDC-0001 for SSMIS, SSM/I, and SMMR.
    # For monthly and aggregate files, `source` is a comman-space-separated string
    # of source filenames.
    source: str,
    # List of satellites that provided data for the given netcdf file.
    platform_ids: list[SUPPORTED_PLATFORM_ID],
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    hemisphere: Hemisphere,
) -> dict[str, Any]:
    """Return a dictionary containing the global attributes for a standard ECDR NetCDF file.

    Note the `date_created` field is populated with the current UTC datetime. It is
    assumed the result of this function will be used to set the attributes of an
    xr.Dataset that will be promptly written to disk as a NetCDF file.
    """
    platforms = [
        PLATFORM_CONFIG.platform_for_id(platform_id) for platform_id in platform_ids
    ]
    platform = ", ".join([platform.name for platform in platforms])
    sensor = ", ".join([platform.sensor for platform in platforms])

    time_coverage_attrs = _get_time_coverage_attrs(
        temporality=temporality,
        aggregate=aggregate,
        time=time,
    )

    if hemisphere == "north":
        keywords = "EARTH SCIENCE > CRYOSPHERE > SEA ICE > SEA ICE CONCENTRATION, Continent > North America > Canada > Hudson Bay, Geographic Region > Arctic, Geographic Region > Polar, Geographic Region > Northern Hemisphere, Ocean > Arctic Ocean, Ocean > Arctic Ocean > Barents Sea, Ocean > Arctic Ocean > Beaufort Sea, Ocean > Arctic Ocean > Chukchi Sea, CONTINENT > NORTH AMERICA > CANADA > HUDSON BAY, Ocean > Atlantic Ocean > North Atlantic Ocean > Davis Straight, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > GULF OF ST LAWRENCE, Ocean > Atlantic Ocean > North Atlantic Ocean > North Sea, Ocean > Atlantic Ocean > North Atlantic Ocean > Norwegian Sea, OCEAN > ATLANTIC OCEAN > NORTH ATLANTIC OCEAN > SVALBARD AND JAN MAYEN, Ocean > Pacific Ocean, Ocean > Pacific Ocean > North Pacific Ocean > Bering Sea, Ocean > Pacific Ocean > North Pacific Ocean > Sea Of Okhotsk"
    else:
        keywords = "EARTH SCIENCE > CRYOSPHERE > SEA ICE > SEA ICE CONCENTRATION, Geographic Region > Polar, Geographic Region > Southern Hemisphere, Ocean > Southern Ocean, Ocean > Southern Ocean > Bellingshausen Sea, Ocean > Southern Ocean > Ross Sea, Ocean > Southern Ocean > Weddell Sea"

    new_global_attrs = OrderedDict(
        # We expect conventions to be the first item in the global attributes.
        Conventions="CF-1.11, ACDD-1.3",
        # Use the current UTC time to set the `date_created` attribute.
        date_created=dt.datetime.utcnow().strftime(DATE_STR_FMT),
        **time_coverage_attrs,
        title=(
            "NOAA-NSIDC Climate Data Record of Passive Microwave"
            f" Sea Ice Concentration Version {ECDR_PRODUCT_VERSION.major_version_number}"
        ),
        program="NOAA Climate Data Record Program",
        software_version_id=_get_software_version_id(),
        metadata_link=f"https://nsidc.org/data/g02202/versions/{ECDR_PRODUCT_VERSION.major_version_number}",
        product_version=ECDR_PRODUCT_VERSION.version_str,
        spatial_resolution=f"{resolution}km",
        standard_name_vocabulary="CF Standard Name Table (v83, 17 October 2023)",
        id="https://doi.org/10.7265/rjzb-pf78",
        naming_authority="org.doi.dx",
        license="No constraints on data access or use",
        summary=f"This data set provides a passive microwave sea ice concentration climate data record (CDR) based on gridded brightness temperatures (TBs) from the Nimbus-7 Scanning Multichannel Microwave Radiometer (SMMR) and the Defense Meteorological Satellite Program (DMSP) series of passive microwave radiometers: the Special Sensor Microwave Imager (SSM/I) and the Special Sensor Microwave Imager/Sounder (SSMIS). The sea ice concentration CDR is an estimate of sea ice concentration that is produced by combining concentration estimates from two algorithms developed at the NASA Goddard Space Flight Center (GSFC): the NASA Team (NT) algorithm and the Bootstrap (BT) algorithm. The individual algorithms are used to process and combine brightness temperature data at NSIDC. This product is designed to provide a consistent time series of sea ice concentrations (the fraction, or percentage, of ocean area covered by sea ice) from November 1978 to the present, which spans the coverage of several passive microwave instruments. The data are gridded on the NSIDC polar stereographic grid with {resolution} km x {resolution} km grid cells and are available in NetCDF file format. Each file contains a variable with the CDR concentration values as well as variables that hold the raw NT and BT processed concentrations for reference. Variables containing standard deviation, quality flags, and projection information are also included. Files that are from 2013 to the present also contain a prototype CDR sea ice concentration based on gridded TBs from the Advanced Microwave Scanning Radiometer 2 (AMSR2) onboard the GCOM-W1 satellite.",
        keywords=keywords,
        keywords_vocabulary="NASA Global Change Master Directory (GCMD) Keywords, Version 17.1",
        cdm_data_type="Grid",
        project="NOAA/NSIDC passive microwave sea ice concentration climate data record",
        creator_name="NSIDC/NOAA",
        creator_url="http://nsidc.org/",
        creator_email="nsidc@nsidc.org",
        institution="NSIDC > National Snow and Ice Data Center",
        processing_level="NOAA Level 3",
        contributor_name="Walter N. Meier, Florence Fetterer, Ann Windnagel, J. Scott Stewart, Trey Stafford",
        contributor_role="principal investigator, author, author, software developer, software developer",
        acknowledgment="This project was supported in part by a grant from the NOAA Climate Data Record Program. The NASA Team and Bootstrap sea ice concentration algorithms were developed by Donald J. Cavalieri, Josefino C. Comiso, Claire L. Parkinson, and others at the NASA Goddard Space Flight Center in Greenbelt, MD.",
        references="Comiso, J. C., and F. Nishio. 2008. Trends in the Sea Ice Cover Using Enhanced and Compatible AMSR-E, SSM/I, and SMMR Data. Journal of Geophysical Research 113, C02S07, doi:10.1029/2007JC0043257. ; Comiso, J. C., D. Cavalieri, C. Parkinson, and P. Gloersen. 1997. Passive Microwave Algorithms for Sea Ice Concentrations: A Comparison of Two Techniques. Remote Sensing of the Environment 60(3):357-84. ; Comiso, J. C. 1984. Characteristics of Winter Sea Ice from Satellite Multispectral Microwave Observations. Journal of Geophysical Research 91(C1):975-94. ; Cavalieri, D. J., P. Gloersen, and W. J. Campbell. 1984. Determination of Sea Ice Parameters with the NIMBUS-7 SMMR. Journal of Geophysical Research 89(D4):5355-5369. ; Cavalieri, D. J., C. l. Parkinson, P. Gloersen, J. C. Comiso, and H. J. Zwally. 1999. Deriving Long-term Time Series of Sea Ice Cover from Satellite Passive-Microwave Multisensor Data Sets. Journal of Geophysical Research 104(7): 15,803-15,814 ; Comiso, J.C., R.A. Gersten, L.V. Stock, J. Turner, G.J. Perez, and K. Cho. 2017. Positive Trend in the Antarctic Sea Ice Cover and Associated Changes in Surface Temperature. J. Climate, 30, 2251–2267, doi:10.1175/JCLI-D-16-0408.1",
        source=source,
        platform=platform,
        sensor=sensor,
        cdr_data_type="grid",
    )

    # Set attributes pulled from a grid-based ancillary ds
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    attrs_from_ancillary = (
        "geospatial_bounds",
        "geospatial_bounds_crs",
        "geospatial_x_units",
        "geospatial_y_units",
        "geospatial_x_resolution",
        "geospatial_y_resolution",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
        "geospatial_lat_units",
        "geospatial_lon_units",
    )

    for attr in attrs_from_ancillary:
        try:
            new_global_attrs[attr] = ancillary_ds.attrs[attr]
        except KeyError:
            logger.warning(f"No such global attr in ancillary file: {attr}")

    return new_global_attrs
