from typing import Literal, cast, get_args

from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS

GRID_ID = Literal["psn12.5", "pss12.5", "psn25", "pss25"]


def get_grid_id(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> GRID_ID:
    grid_id = f"ps{hemisphere[0]}{resolution}"
    if grid_id not in get_args(GRID_ID):
        raise RuntimeError(f"Could not determine grid for {hemisphere=} {resolution=}")

    grid_id = cast(GRID_ID, grid_id)

    return grid_id
