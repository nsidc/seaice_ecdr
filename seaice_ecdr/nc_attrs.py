import datetime as dt
import subprocess
from collections import OrderedDict
from functools import cache
from typing import Any, Final, Literal

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


def _get_time_coverage_duration_resolution(
    temporality: Temporality,
    aggregate: bool,
) -> dict[str, Any]:
    """Return a dictionary of time coverage and resolution attrs.

    TODO: verify this function behaves the way we want. See notes below.

    * `time_coverage_duration`:
       For individual daily files this is P1D, for an aggregated file this is
       P1Y if it’s a full year. It looks like if it’s a partial year in V4 we
       do P1Y as well. Should we change that to be PXXD where XX is the
       number of days until we get a full year? For the monthly aggregate, v4
       doesn’t have this attribute at all. We should have it. It could say
       P45Y3M when we have a partial year. Let’s discuss.

    * `time_coverage_resolution`:
       For monthly file this is P1M. For daily, "P1D"
    """
    time_coverage_attrs = {}
    if temporality == "daily":
        time_coverage_attrs["time_coverage_duration"] = "P1D"
        time_coverage_attrs["time_coverage_resolution"] = "P1D"

        if aggregate:
            # TODO: for partial years, do we use the `PXXD` format where `XX` is
            # the number of days until we get a full year?
            time_coverage_attrs["time_coverage_duration"] = "P1Y"
    else:
        time_coverage_attrs["time_coverage_duration"] = "P1M"
        time_coverage_attrs["time_coverage_resolution"] = "P1M"
        if aggregate:
            # TODO: aggregate monthly files don't have a
            # `time_coverage_duration` Should they? We need to discuss.
            time_coverage_attrs.pop("time_coverage_duration")

    return time_coverage_attrs


def get_global_attrs(
    *,
    time_coverage_start: dt.datetime,
    time_coverage_end: dt.datetime,
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
) -> dict[str, Any]:
    """Return a dictionary containing the global attributes for a standard ECDR NetCDF file.

    Note the `date_created` field is populated with the current UTC datetime. It is
    assumed the result of this function will be used to set the attributes of an
    xr.Dataset that will be promptly written to disk as a NetCDF file.
    """

    # TODO: support different resolutions, platforms, and sensors!
    resolution: Final = "12.5"
    # Here’s what the GCMD platform long name should be based on sensor/platform short name:
    # AMRS2: “GCOM-W1 > Global Change Observation Mission 1st-Water”
    # AMRS-E: " Aqua > Earth Observing System, Aqua”
    # SSMIS on F17: “DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17”
    # SSM/I on F13: “DMSP 5D-2/F13 > Defense Meteorological Satellite Program-F13”
    # SSM/I on F11: “DMSP 5D-2/F11 > Defense Meteorological Satellite Program-F11”
    # SSM/I on F8: “DMSP 5D-2/F8 > Defense Meteorological Satellite Program-F8”
    # SMMR: “Nimbus-7”
    platform: Final = "GCOM-W1 > Global Change Observation Mission 1st-Water"
    # Here’s what the GCMD sensor name should be based on sensor short name:
    # AMRS2: “AMSR2 > Advanced Microwave Scanning Radiometer 2”
    # AMRS-E: “AMSR-E > Advanced Microwave Scanning Radiometer-EOS”
    # SSMIS: “SSMIS > Special Sensor Microwave Imager/Sounder”
    # SSM/I: “SSM/I > Special Sensor Microwave/Imager”
    # SMMR: “SMMR > Scanning Multichannel Microwave Radiometer”
    sensor: Final = "AMSR2 > Advanced Microwave Scanning Radiometer 2"

    time_coverage_attrs = _get_time_coverage_duration_resolution(
        temporality=temporality,
        aggregate=aggregate,
    )

    new_global_attrs = OrderedDict(
        # We expect conventions to be the first item in the global attributes.
        conventions="CF-1.10, ACDD-1.3",
        # Use the current UTC time to set the `date_created` attribute.
        date_created=dt.datetime.utcnow().strftime(DATE_STR_FMT),
        time_coverage_start=time_coverage_start.strftime(DATE_STR_FMT),
        time_coverage_end=time_coverage_end.strftime(DATE_STR_FMT),
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
