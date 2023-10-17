import numpy as np
import numpy.typing as npt


def psn_125_near_pole_hole_mask() -> npt.NDArray[np.bool_]:
    """Return a mask of the area near the pole hole for the psn125 grid.

    Identify pole hole pixels for psn12.5
    These pixels were identified by examining AUSI12-derived NH fields in 2021
       and are one ortho and diag from the commonly no-data pixels near
       the pole that year from AU_SI12 products
    """
    pole_pixels = np.zeros((896, 608), dtype=np.uint8)
    pole_pixels[461, 304 : 311 + 1] = 1
    pole_pixels[462, 303 : 312 + 1] = 1
    pole_pixels[463, 302 : 313 + 1] = 1
    pole_pixels[464 : 471 + 1, 301 : 314 + 1] = 1
    pole_pixels[472, 302 : 313 + 1] = 1
    pole_pixels[473, 303 : 312 + 1] = 1
    pole_pixels[474, 304 : 311 + 1] = 1

    pole_pixels_bool = pole_pixels.astype(bool)

    return pole_pixels_bool
