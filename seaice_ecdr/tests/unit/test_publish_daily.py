import xarray as xr
from datatree import DataTree

from seaice_ecdr import publish_daily


def test__remove_valid_range_from_coord_vars():
    mock_ds = DataTree.from_dict(
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
    publish_daily._remove_valid_range_from_coord_vars(mock_ds)
    assert "valid_range" not in mock_ds.x.attrs.keys()
    assert "valid_range" not in mock_ds.y.attrs.keys()
    assert "valid_range" not in mock_ds.time.attrs.keys()


def test__add_coordinate_coverage_type():
    mock_ds = DataTree.from_dict(
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
    publish_daily._add_coordinate_coverage_content_type(mock_ds)
    assert "coverage_content_type" in mock_ds.x.attrs.keys()
    assert "coverage_content_type" in mock_ds.y.attrs.keys()
    assert "coverage_content_type" in mock_ds.time.attrs.keys()
