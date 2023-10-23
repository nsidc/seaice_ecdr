import numpy.testing as npt
import numpy as np

from seaice_ecdr import melt


def test_with_melting_everywhere():
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([200, 200, 200, 200])
    tb37 = np.array([200, 200, 200, 200])
    expected = np.array([True, True, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_with_some_non_melting_due_to_large_tb_deltas():
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([200, 200, 200, 200])
    tb37 = np.array([150, 200, 200, 200])
    expected = np.array([False, True, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_that_tb_threshold_is_2_kelvin():
    # Brightness temperatures are in *tenths* of kelvin
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([200, 200, 200, 200])
    tb37 = np.array([150, 180, 181, 200])
    expected = np.array([False, False, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_with_some_non_melting_due_to_low_concentrations():
    concentrations = np.array([10, 40, 80, 90])
    tb19 = np.array([200, 200, 200, 200])
    tb37 = np.array([200, 200, 200, 200])
    expected = np.array([False, False, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_that_50_percent_concentration_is_in_range():
    concentrations = np.array([49, 50, 80, 90])
    tb19 = np.array([200, 200, 200, 200])
    tb37 = np.array([200, 200, 200, 200])
    expected = np.array([False, True, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_with_missing_tb19_values():
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([melt.TB_MISSING, melt.TB_MISSING, 200, 200])
    tb37 = np.array([200, 200, 200, 200])
    expected = np.array([False, False, True, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_with_missing_tb37_values():
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([200, melt.TB_MISSING + 1, melt.TB_MISSING + 1, 200])
    tb37 = np.array([200, melt.TB_MISSING, melt.TB_MISSING, 200])
    expected = np.array([True, False, False, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)


def test_with_both_missing_values():
    concentrations = np.array([80, 80, 80, 90])
    tb19 = np.array([200, melt.TB_MISSING, melt.TB_MISSING, 200])
    tb37 = np.array([200, melt.TB_MISSING, melt.TB_MISSING, 200])
    expected = np.array([True, False, False, True])

    actual = melt.melting(concentrations, tb19, tb37)

    npt.assert_array_equal(expected, actual)
