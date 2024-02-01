import datetime as dt
import itertools
from typing import get_args

import pytest
from pm_tb_data._types import Hemisphere

from seaice_ecdr.tests.integration import ecdr_data_dir_test_path  # noqa
from seaice_ecdr.validation import validate_outputs


@pytest.mark.order(after="test_monthly.py::test_make_monthly_nc")
def test_validate_outputs(ecdr_data_dir_test_path):  # noqa
    for product_type, hemisphere in itertools.product(
        ("daily", "monthly"), get_args(Hemisphere)
    ):
        outputs = validate_outputs(
            hemisphere=hemisphere,
            start_date=dt.date(2022, 3, 1),
            end_date=dt.date(2022, 3, 4),
            # Sometimes, mypy wants: # type:ignore[arg-type]
            # product=product_type,
            product=product_type,
            ecdr_data_dir=ecdr_data_dir_test_path,
        )

        assert outputs["error_filepath"].is_file()
        assert outputs["log_filepath"].is_file()

        # TODO: uncomment? The assertion isn't true in all cases (currently)
        # because we do no thresholding of conc values but we consider anything
        # less than 10% to be "bad", which results in an error code of -999.
        # with open(outputs["error_filepath"], "r") as error_csv:
        #     reader = csv.DictReader(error_csv)
        #     for row in reader:
        #         assert row["error_code"] == 0
