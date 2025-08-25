import xarray as xr

from seaice_ecdr.nc_util import (
    add_coordinate_coverage_content_type,
    remove_valid_range_from_coordinate_vars,
)


def test_remove_valid_range_from_coord_vars():
    mock_ds = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                {
                    "time": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                    "x": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                    "y": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                },
            ),
        }
    )

    assert "valid_range" in mock_ds.x.attrs.keys()
    remove_valid_range_from_coordinate_vars(mock_ds)
    assert "valid_range" not in mock_ds.x.attrs.keys()
    assert "valid_range" not in mock_ds.y.attrs.keys()
    assert "valid_range" not in mock_ds.time.attrs.keys()


def test_add_coordinate_coverage_type():
    mock_ds = xr.DataTree.from_dict(
        {
            "/": xr.Dataset(
                {
                    "time": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                    "x": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                    "y": xr.DataArray([1, 2, 3], attrs={"valid_range": (0, 4)}),
                },
            ),
        }
    )

    assert "coverage_content_type" not in mock_ds.x.attrs.keys()
    add_coordinate_coverage_content_type(mock_ds)
    assert "coverage_content_type" in mock_ds.x.attrs.keys()
    assert "coverage_content_type" in mock_ds.y.attrs.keys()
    assert "coverage_content_type" in mock_ds.time.attrs.keys()
