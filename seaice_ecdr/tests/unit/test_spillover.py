"""Tests of the land spillover routines coded in seaice_ecdr.py.  """

import numpy as np
import numpy.testing as nptesting

from seaice_ecdr import spillover

test_ils_array = np.array(
    [
        [4, 3, 3, 2, 2, 2, 1, 1, 1],
        [4, 3, 2, 2, 2, 1, 1, 1, 1],
        [4, 3, 2, 2, 1, 1, 1, 2, 1],  # includes isolated "coast" cell
        [4, 3, 2, 2, 2, 2, 1, 1, 1],
        [4, 3, 2, 2, 1, 2, 2, 2, 2],
        [4, 3, 2, 2, 1, 1, 1, 2, 1],
        [4, 3, 2, 2, 1, 1, 1, 1, 2],  # includes diag-conn inlet cell
    ],
    dtype=np.uint8,
)


def test_ils_algorithm_requires_valid_arrvalues(tmpdir):
    """ILS algorithm should check for valid ils_array."""
    ils_array = test_ils_array.copy()
    ils_array[:, 0] = 5

    init_conc = test_ils_array.copy().astype(np.float32)
    init_conc[test_ils_array > 1] = 1.0
    expected_conc = init_conc.copy()

    try:
        filtered_conc = spillover.improved_land_spillover(
            ils_arr=ils_array,
            init_conc=init_conc,
        )
    except AssertionError:
        # We expect an AssertionError
        return

    # Satisfy vulture...
    assert expected_conc is not None
    assert filtered_conc is not None

    raise ValueError("We expected ils_array to fail with values of 5")


def test_ils_algorithm_keeps_anchored_ice(tmpdir):
    """ILS algorithm should not delete anything if all values are high conc."""
    init_conc = test_ils_array.copy().astype(np.float32)
    init_conc[test_ils_array > 1] = 1.0
    expected_conc = init_conc.copy()

    filtered_conc = spillover.improved_land_spillover(
        ils_arr=test_ils_array,
        init_conc=init_conc,
    )

    nptesting.assert_array_equal(filtered_conc, expected_conc)


def test_ils_algorithm_removes_unanchored_ice(tmpdir):
    """ILS algorithm should not delete anything if all values are high conc."""
    init_conc = test_ils_array.copy().astype(np.float32)
    init_conc[test_ils_array > 1] = 1.0
    init_conc[test_ils_array == 3] = 0.0
    expected_conc = init_conc.copy()
    expected_conc[test_ils_array == 2] = 0.0

    filtered_conc = spillover.improved_land_spillover(
        ils_arr=test_ils_array,
        init_conc=init_conc,
    )

    breakpoint()
    nptesting.assert_array_equal(filtered_conc, expected_conc)
