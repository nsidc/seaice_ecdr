from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pm_icecon.land_spillover import apply_nt2_land_spillover
from pm_icecon.nt.compute_nt_ic import apply_nt_spillover
from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import (
    get_adj123_field,
    get_land90_conc_field,
)
from seaice_ecdr.tb_data import (
    EcdrTbData,
)
from seaice_ecdr.util import get_ecdr_grid_shape

NT_MAPS_DIR = Path("/share/apps/G02202_V5/cdr_testdata/nt_datafiles/data36/maps")


def _get_25km_shoremap(*, hemisphere: Hemisphere):
    shoremap_fn = NT_MAPS_DIR / f"shoremap_{hemisphere}_25"
    shoremap = np.fromfile(shoremap_fn, dtype=">i2")[150:].reshape(
        get_ecdr_grid_shape(hemisphere=hemisphere, resolution="25")
    )

    return shoremap


def _get_25km_minic(*, hemisphere: Hemisphere):
    if hemisphere == "north":
        minic_fn = "SSMI8_monavg_min_con"
    else:
        minic_fn = "SSMI_monavg_min_con_s"

    minic_path = NT_MAPS_DIR / minic_fn
    minic = np.fromfile(minic_path, dtype=">i2")[150:].reshape(
        get_ecdr_grid_shape(hemisphere=hemisphere, resolution="25")
    )

    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    minic = minic / 10

    return minic


def land_spillover(
    *,
    cdr_conc: npt.NDArray,
    hemisphere: Hemisphere,
    tb_data: EcdrTbData,
    algorithm: Literal["NT2", "NT"],
) -> npt.NDArray:
    """Apply the land spillover technique to the CDR concentration field."""
    if algorithm == "NT2":
        l90c = get_land90_conc_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
        )
        adj123 = get_adj123_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
        )
        spillover_applied = apply_nt2_land_spillover(
            conc=cdr_conc,
            adj123=adj123.data,
            l90c=l90c.data,
            anchoring_siconc=50.0,
            affect_dist3=True,
        )
    elif algorithm == "NT":
        shoremap = _get_25km_shoremap(hemisphere=hemisphere)
        minic = _get_25km_minic(hemisphere=hemisphere)
        spillover_applied = apply_nt_spillover(
            conc=cdr_conc,
            shoremap=shoremap,
            minic=minic,
        )
    else:
        raise RuntimeError(
            f"The requested land spillover algorithm ({algorithm=}) is not implemented."
        )

    return spillover_applied
