import csv
import datetime as dt
import itertools

import pytest
from pm_tb_data._types import NORTH

from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa
from seaice_ecdr.validation import validate_outputs


@pytest.mark.skip(
    reason="skipping because currently adding CDRv4 flags to conc fields in order to match CDRv4 results, but in CDRv5 we want all non-valid values to be the fill value"
)
@pytest.mark.order(after="test_monthly.py::test_make_intermediate_monthly_nc")
def test_validate_outputs(base_output_dir_test_path):  # noqa
    for product_type, hemisphere in itertools.product(
        # TODO: currently just iterate over NORTH, because that's what data is
        # created for in the depdendent tests. We may consider creating monthly
        # data too.
        ("daily", "monthly"),
        (NORTH,),
    ):
        outputs = validate_outputs(
            hemisphere=hemisphere,
            start_date=dt.date(2022, 3, 1),
            end_date=dt.date(2022, 3, 4),
            product=product_type,  # type: ignore[arg-type]
            base_output_dir=base_output_dir_test_path,
        )

        assert outputs["error_filepath"].is_file()
        assert outputs["log_filepath"].is_file()

        # TODO: daily data currently produces an error code due to SIC values
        # <10%. We think this arises from temporal interpolation of the daily
        # fields.
        if product_type == "monthly":
            with open(outputs["error_filepath"], "r") as error_csv:
                reader = csv.DictReader(error_csv)
                for row in reader:
                    try:
                        assert row["error_code"] == "0"
                    except AssertionError as e:
                        print(f'nonzero error_code: {row["error_code"]}')
                        raise e
