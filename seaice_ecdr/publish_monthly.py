import datetime as dt
from functools import cache
from pathlib import Path

import datatree
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.intermediate_monthly import (
    get_intermediate_monthly_dir,
)
from seaice_ecdr.nc_util import (
    add_coordinate_coverage_content_type,
    add_coordinates_attr,
    fix_monthly_ncattrs,
    remove_valid_range_from_coordinate_vars,
)
from seaice_ecdr.nrt import override_attrs_for_nrt
from seaice_ecdr.platforms import SUPPORTED_PLATFORM_ID
from seaice_ecdr.util import (
    find_standard_monthly_netcdf_files,
    get_complete_output_dir,
    get_intermediate_output_dir,
    nrt_monthly_filename,
    platform_id_from_filename,
    standard_monthly_filename,
)

# TODO: consider extracting to config or a kwarg of this function for more
# flexible use with other platforms in the future.
PROTOTYPE_PLATFORM_ID: SUPPORTED_PLATFORM_ID = "am2"
PROTOTYPE_PLATFORM_DATA_GROUP_NAME = f"prototype_{PROTOTYPE_PLATFORM_ID}"
# TODO: this should be extracted from e.g., the platform start date
# configuration instead of hard-coding it here.
PROTOTYPE_PLATFORM_START_DATE = dt.date(2013, 1, 1)


def get_complete_monthly_dir(complete_output_dir: Path) -> Path:
    monthly_dir = complete_output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    return monthly_dir


def get_complete_monthly_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID,
    year: int,
    month: int,
    complete_output_dir: Path,
    is_nrt: bool,
) -> Path:
    output_dir = get_complete_monthly_dir(
        complete_output_dir=complete_output_dir,
    )

    if is_nrt:
        output_fn = nrt_monthly_filename(
            hemisphere=hemisphere,
            resolution=resolution,
            platform_id=platform_id,
            year=year,
            month=month,
        )
    else:
        output_fn = standard_monthly_filename(
            hemisphere=hemisphere,
            resolution=resolution,
            platform_id=platform_id,
            year=year,
            month=month,
        )

    output_path = output_dir / output_fn

    return output_path


@cache
def _get_all_intermediate_monthly_fps(
    *,
    base_output_dir: Path,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> list[Path]:
    """Get a list of all intermediate monthly filepaths for the given params."""
    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    intermediate_monthly_dir = get_intermediate_monthly_dir(
        intermediate_output_dir=intermediate_output_dir,
    )

    all_intermediate_monthly_fps = find_standard_monthly_netcdf_files(
        search_dir=intermediate_monthly_dir,
        hemisphere=hemisphere,
        resolution=resolution,
        year=year,
        month=month,
        platform_id="*",
    )
    return all_intermediate_monthly_fps


def _get_intermediate_monthly_fp(
    *,
    base_output_dir: Path,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> Path:
    all_intermediate_monthly_fps = _get_all_intermediate_monthly_fps(
        base_output_dir=base_output_dir,
        year=year,
        month=month,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    intermediate_monthly_fps = [
        fp
        for fp in all_intermediate_monthly_fps
        if f"_{PROTOTYPE_PLATFORM_ID}_" not in fp.name
    ]
    if len(intermediate_monthly_fps) != 1:
        raise FileNotFoundError(
            f"Failed to find an intermediate monthly file for {year=}, {month=}, {hemisphere}"
        )

    monthly_filepath = intermediate_monthly_fps[0]

    return monthly_filepath


def _get_prototype_monthly_fp(
    *,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> Path | None:
    all_intermediate_monthly_fps = _get_all_intermediate_monthly_fps(
        base_output_dir=base_output_dir,
        year=year,
        month=month,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    prototype_monthly_fps = [
        fp
        for fp in all_intermediate_monthly_fps
        if f"_{PROTOTYPE_PLATFORM_ID}_" in fp.name
    ]

    if (prototype_len := len(prototype_monthly_fps)) > 0:
        if prototype_len > 1:
            raise RuntimeError(
                f"Something went wrong: found multiple intermediate prototype monthly files, but found {prototype_len}: {prototype_monthly_fps}."
            )

        return prototype_monthly_fps[0]
    else:
        return None


def prepare_monthly_ds_for_publication(
    *,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    intermediate_monthly_fp: Path,
    prototype_monthly_fp: Path | None,
) -> datatree.DataTree:
    # Get the intermediate monthly data
    default_intermediate_monthly_ds = xr.open_dataset(intermediate_monthly_fp)

    # TODO: a lot of the below (e.g., where the supplementary group is
    # constructed and a DataTree) is made is very similar to how it's done in
    # `publish_daily`. DRY this out.

    # Publication-ready monthly data are grouped using `DataTree`. Create a
    # `cdr_supplementary` group for "supplementary" fields:
    cdr_supplementary_fields = [
        "surface_type_mask",
    ]
    # Melt onset only occurs in the NH.
    if hemisphere == NORTH:
        cdr_supplementary_fields.append("cdr_melt_onset_day_monthly")
    cdr_supplementary_group = default_intermediate_monthly_ds[cdr_supplementary_fields]
    # Drop x, y, time coordinate variables. These will be inherited from the
    # root group.
    cdr_supplementary_group = cdr_supplementary_group.drop_vars(["x", "y", "time"])
    # remove attrs from supplementary group. These will be inherted from the
    # root group.
    cdr_supplementary_group.attrs = {}

    # TODO
    complete_monthly_ds: datatree.DataTree = datatree.DataTree.from_dict(
        {
            "/": default_intermediate_monthly_ds[
                [
                    k
                    for k in default_intermediate_monthly_ds
                    if k not in cdr_supplementary_fields
                ]
            ],
            "cdr_supplementary": cdr_supplementary_group,
        }
    )

    # Add the prototype group if a prototype file is passed in.
    if prototype_monthly_fp:
        prototype_monthly_ds = xr.open_dataset(prototype_monthly_fp)
        cdr_var_fieldnames = [
            "cdr_seaice_conc_monthly",
            "cdr_seaice_conc_monthly_qa_flag",
            "cdr_seaice_conc_monthly_stdev",
        ]
        remap_names = {
            cdr_var_fieldname: cdr_var_fieldname.replace(
                "cdr_", f"{PROTOTYPE_PLATFORM_ID}_"
            )
            for cdr_var_fieldname in cdr_var_fieldnames
        }

        prototype_subgroup = prototype_monthly_ds[cdr_var_fieldnames].rename_vars(
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
            if k in ["sensor", "platform"]
        }

        complete_monthly_ds[PROTOTYPE_PLATFORM_DATA_GROUP_NAME] = datatree.DataTree(
            data=prototype_subgroup,
        )
    elif dt.date(year, month, 1) >= PROTOTYPE_PLATFORM_START_DATE:
        logger.warning(
            f"Failed to find prototype monthly file for {year=} {month=} {PROTOTYPE_PLATFORM_ID=}"
        )

    # Do final cleanup. This should be unnecessary (see the docstrings for the
    # associated fuctions for more).
    remove_valid_range_from_coordinate_vars(complete_monthly_ds)
    add_coordinate_coverage_content_type(complete_monthly_ds)
    add_coordinates_attr(complete_monthly_ds)

    return complete_monthly_ds


def _write_publication_ready_nc_and_checksum(
    publication_ready_monthly_ds: datatree.DataTree,
    base_output_dir: Path,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    is_nrt: bool,
    platform_id: SUPPORTED_PLATFORM_ID,
) -> Path:
    # Write out finalized nc file.
    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )

    complete_monthly_filepath = get_complete_monthly_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        year=year,
        month=month,
        complete_output_dir=complete_output_dir,
        is_nrt=is_nrt,
    )

    # Ensure consistency of time units
    publication_ready_monthly_ds.time.encoding["units"] = "days since 1970-01-01"
    publication_ready_monthly_ds.time.encoding["calendar"] = "standard"

    publication_ready_monthly_ds.to_netcdf(complete_monthly_filepath)
    logger.success(f"Staged NC file for publication: {complete_monthly_filepath}")

    # Write checksum file for the complete daily output.
    write_checksum_file(
        input_filepath=complete_monthly_filepath,
        output_dir=complete_output_dir / "checksums" / "monthly",
    )

    return complete_monthly_filepath


def prepare_monthly_nc_for_publication(
    *,
    base_output_dir: Path,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    is_nrt: bool,
):
    """Prepare a monthly NetCDF file for publication.

    If monthly data for a prototype platform is available for the given month,
    this function adds that data to a prototype group in the output NetCDF
    file. The output NC file's root-group variables are all taken from the
    default platforms given by the platofrm start date configuration.
    """
    intermediate_monthly_fp = _get_intermediate_monthly_fp(
        base_output_dir=base_output_dir,
        year=year,
        month=month,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # Now get the prototype filepath, if it exists, and add it to the new
    # monthly ds.
    prototype_monthly_fp = _get_prototype_monthly_fp(
        year=year,
        month=month,
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
        resolution=resolution,
    )
    complete_monthly_ds = prepare_monthly_ds_for_publication(
        year=year,
        month=month,
        hemisphere=hemisphere,
        intermediate_monthly_fp=intermediate_monthly_fp,
        prototype_monthly_fp=prototype_monthly_fp,
    )

    # Fix monthly nc-attributes
    fix_monthly_ncattrs(complete_monthly_ds)

    # Override attrs for nrt
    if is_nrt:
        complete_monthly_ds = override_attrs_for_nrt(
            publication_ready_ds=complete_monthly_ds, resolution=resolution
        )

    # get the platform Id from the filename default filename.
    platform_id = platform_id_from_filename(intermediate_monthly_fp.name)
    # Write the publication-ready monthly ds
    complete_monthly_filepath = _write_publication_ready_nc_and_checksum(
        publication_ready_monthly_ds=complete_monthly_ds,
        year=year,
        month=month,
        hemisphere=hemisphere,
        resolution=resolution,
        base_output_dir=base_output_dir,
        platform_id=platform_id,
        is_nrt=is_nrt,
    )

    return complete_monthly_filepath
