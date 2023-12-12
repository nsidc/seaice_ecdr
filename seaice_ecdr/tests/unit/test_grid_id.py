from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.grid_id import get_grid_id


def test_get_grid_id():
    assert "psn12.5" == get_grid_id(hemisphere=NORTH, resolution="12.5")
    assert "pss12.5" == get_grid_id(hemisphere=SOUTH, resolution="12.5")
