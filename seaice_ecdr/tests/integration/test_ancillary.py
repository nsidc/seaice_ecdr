from seaice_ecdr import ancillary


def test_verify_invalid_ice_mask_is_boolean():  # noqa
    """If the dtype attribute of invalid_ice_mask isn't set to bool,
    then using it to mask data in xarray will fail."""
    invalid_ice_mask = ancillary.get_invalid_ice_mask(
        hemisphere="north",
        month=11,
        resolution="12.5",
    )
    assert invalid_ice_mask.dtype == "bool"
