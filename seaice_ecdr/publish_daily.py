import datetime as dt
from pathlib import Path

from datatree import DataTree
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.complete_daily_ecdr import read_cdecdr_ds
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID
from seaice_ecdr.util import (
    get_complete_output_dir,
    get_intermediate_output_dir,
    nrt_daily_filename,
    standard_daily_filename,
)


# TODO: this and `get_complete_daily_filepath` are identical (aside from var
# names) to `get_ecdr_filepath` and `get_ecdr_dir`.
def get_complete_daily_dir(
    *,
    complete_output_dir: Path,
    year: int,
    # TODO: extract nrt handling and make responsiblity for defining the output
    # dir a higher-level concern.
    is_nrt: bool,
) -> Path:
    if is_nrt:
        # NRT daily data just lives under the complete output dir.
        ecdr_dir = complete_output_dir
    else:
        ecdr_dir = complete_output_dir / "daily" / str(year)
    ecdr_dir.mkdir(parents=True, exist_ok=True)

    return ecdr_dir


def get_complete_daily_filepath(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    complete_output_dir: Path,
    is_nrt: bool,
):
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    if is_nrt:
        ecdr_filename = nrt_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform.id,
            resolution=resolution,
        )
    else:
        ecdr_filename = standard_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform.id,
            resolution=resolution,
        )

    ecdr_dir = get_complete_daily_dir(
        complete_output_dir=complete_output_dir,
        year=date.year,
        is_nrt=is_nrt,
    )

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath


def publish_daily_nc(
    *,
    base_output_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> None:
    """Prepares a daily nc file for publication.

    If data for the provided prototype platform is available for the given date,
    this function adds that data to a prototype group in the output NC file. The
    output NC file's root-group variables are all taken from the default
    platforms given by the platform start date configuration.
    """
    # TODO: consider extracting to config or a kwarg of this function for more
    # flexible use with other platforms in the future.
    PROTOTYPE_PLATFORM_ID: SUPPORTED_PLATFORM_ID = "am2"
    PROTOTYPE_PLATFORM_DATA_GROUP_NAME = "prototype_amsr2"

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )
    default_platform = PLATFORM_CONFIG.get_platform_by_date(date)
    default_daily_ds = read_cdecdr_ds(
        intermediate_output_dir=intermediate_output_dir,
        resolution=resolution,
        date=date,
        platform_id=default_platform.id,
        hemisphere=hemisphere,
        is_nrt=False,
    )

    # publication-ready daily data are grouped using `DataTree`.
    # Create a `cdr_supplementary` group for "supplemntary" fields
    cdr_supplementary_fields = [
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "surface_type_mask",
    ]
    # Melt onset only occurs in the NH.
    if hemisphere == NORTH:
        cdr_supplementary_fields.append("cdr_melt_onset_day")

    # Drop x, y, time coordinate variables. These will be inherited from the
    # root group.
    cdr_supplementary_group = default_daily_ds[cdr_supplementary_fields].drop_vars(
        ["x", "y", "time"]
    )
    # remove attrs from supplementary group. These will be inherted from the
    # root group.
    cdr_supplementary_group.attrs = {}

    complete_daily_ds: DataTree = DataTree.from_dict(
        {
            "/": default_daily_ds[
                [k for k in default_daily_ds if k not in cdr_supplementary_fields]
            ],
            "cdr_supplementary": cdr_supplementary_group,
        }
    )

    if PLATFORM_CONFIG.platform_available_for_date(
        platform_id=PROTOTYPE_PLATFORM_ID,
        date=date,
    ):
        try:
            prototype_daily_ds = read_cdecdr_ds(
                intermediate_output_dir=intermediate_output_dir,
                resolution=resolution,
                date=date,
                platform_id=PROTOTYPE_PLATFORM_ID,
                hemisphere=hemisphere,
                is_nrt=False,
            )
            cdr_var_fieldnames = [
                "cdr_seaice_conc",
                "cdr_qa_seaice_conc",
                "cdr_spatial_interpolation_flag",
                "cdr_temporal_interpolation_flag",
                "cdr_stdev_seaice_conc",
            ]
            remap_names = {
                cdr_var_fieldname: cdr_var_fieldname.replace(
                    "cdr_", f"{PROTOTYPE_PLATFORM_ID}_"
                )
                for cdr_var_fieldname in cdr_var_fieldnames
            }
            prototype_subgroup = prototype_daily_ds[cdr_var_fieldnames].rename_vars(
                remap_names
            )
            # Rename ancillary variables.
            for var in prototype_subgroup.values():
                if "ancillary_variables" in var.attrs:
                    var.attrs["ancillary_variables"] = var.attrs[
                        "ancillary_variables"
                    ].replace("cdr_", f"{PROTOTYPE_PLATFORM_ID}_")
            # Drop x, y, and time coordinate variables. These will be inherited from the parent.
            prototype_subgroup = prototype_subgroup.drop_vars(["x", "y", "time"])
            # Retain only the group-specific global attrs
            prototype_subgroup.attrs = {
                k: v
                for k, v in prototype_subgroup.attrs.items()
                if k in ["source", "sensor", "platform"]
            }
            complete_daily_ds[PROTOTYPE_PLATFORM_DATA_GROUP_NAME] = DataTree(
                data=prototype_subgroup,
            )
        except FileNotFoundError:
            logger.warning(
                f"Failed to find prototype daily file for {date=} {PROTOTYPE_PLATFORM_ID=}"
            )

    # write out finalized nc file.
    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )
    complete_daily_filepath = get_complete_daily_filepath(
        date=date,
        resolution=resolution,
        complete_output_dir=complete_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )
    complete_daily_ds.to_netcdf(complete_daily_filepath)


if __name__ == "__main__":
    publish_daily_nc(
        base_output_dir=Path("/share/apps/G02202_V5/25km/combined/"),
        hemisphere="north",
        resolution="25",
        date=dt.date(2022, 3, 2),
    )
