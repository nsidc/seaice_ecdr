"""Pydantic data models for platform configuration."""

import copy
import datetime as dt
from typing import Literal, cast

from pydantic import BaseModel, root_validator, validator

# TODO: ideally this is used sparingly. The code should accept any number of
# platform configurations, and those configurations should defined what's
# "supported". We could even include the import string for a fetch function that
# conforms to a spec for each platform, so that the e.g., `tb_data` module does
# not need to map specific IDs to functions. See:
# https://docs.pydantic.dev/2.3/usage/types/string_types/#importstring
SUPPORTED_PLATFORM_ID = Literal[
    "am2",  # AMSR2
    "ame",  # AMSRE
    "F18",  # SSMIS F18 (NRT)
    "F17",  # SSMIS F17
    "F13",  # SSMI F13
    "F11",  # SSMI F11
    "F08",  # SSMI F08
    "n07",  # Nimbus-7 SMMR
]


class DateRange(BaseModel):
    first_date: dt.date
    # If the last_date is None, it indicates that the satellite is still
    # operating and we do not have a "last date" yet.
    last_date: dt.date | None

    @root_validator(skip_on_failure=True)
    def validate_date_range(
        cls,  # noqa: F841 (`cls` is unused, but must be present for pydantic)
        values,
    ):
        first_date: dt.date = values["first_date"]
        last_date: dt.date | None = values["last_date"]

        # If the last date isn't given, it means date range extends from the
        # first date into the future (satellite is still operating)
        if (last_date is not None) and (first_date > last_date):
            raise ValueError(
                f"First date ({first_date}) is after last date {last_date} in date range."
            )

        return values


class Platform(BaseModel):
    # E.g., "DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17"
    name: str
    # GCMD sensor name. E.g., SSMIS > Special Sensor Microwave Imager/Sounder
    sensor: str
    # E.g., "F17"
    id: SUPPORTED_PLATFORM_ID
    # The available date range for the platform, inclusive.
    date_range: DateRange


def platform_for_id(
    *, platforms: list[Platform], platform_id: SUPPORTED_PLATFORM_ID
) -> Platform:
    for platform in platforms:
        if platform_id == platform.id:
            return platform
    raise ValueError(f"Could not find platform with id {platform_id}.")


class PlatformStartDate(BaseModel):
    platform_id: SUPPORTED_PLATFORM_ID
    start_date: dt.date


class PlatformConfig(BaseModel):
    platforms: list[Platform]
    cdr_platform_start_dates: list[PlatformStartDate]

    @root_validator(skip_on_failure=True)
    def validate_platform_start_dates_platform_in_platforms(
        cls,  # noqa: F841 (`cls` is unused, but must be present for pydantic)
        values,
    ):
        """Validate that each platform start date corresponds with a defined platform."""
        platform_ids = [platform.id for platform in values["platforms"]]
        for platform_start_date in values["cdr_platform_start_dates"]:
            if platform_start_date.platform_id not in platform_ids:
                raise ValueError(
                    f"Did not find {platform_start_date.platform_id} in platform list (must be one of {platform_ids})."
                )

        return values

    @root_validator(skip_on_failure=True)
    def validate_platform_start_date_in_platform_date_range(
        cls,  # noqa: F841 (`cls` is unused, but must be present for pydantic)
        values,
    ):
        """Validate that each platform start date is within the platform's date range."""
        for platform_start_date in values["cdr_platform_start_dates"]:
            matching_platform = platform_for_id(
                platforms=values["platforms"],
                platform_id=platform_start_date.platform_id,
            )
            start_date_before_first_date = (
                matching_platform.date_range.first_date > platform_start_date.start_date
            )

            last_date_is_not_none = matching_platform.date_range.last_date is not None
            start_date_after_last_date = last_date_is_not_none and (
                matching_platform.date_range.last_date < platform_start_date.start_date
            )

            if start_date_before_first_date or start_date_after_last_date:
                raise ValueError(
                    f"Platform start date of {platform_start_date.start_date}"
                    f" for {matching_platform.id}"
                    " is outside of the platform's date range:"
                    f" {matching_platform.date_range}"
                )
        return values

    @validator("cdr_platform_start_dates")
    def validate_platform_start_dates_in_order(
        cls,  # noqa: F841 (`cls` is unused, but must be present for pydantic)
        values: list[PlatformStartDate],
    ) -> list[PlatformStartDate]:
        """Validate that platform start dates are defined in order from old -> new.

        E.g., 1979-10-25 should be listed before 1987-07-10.
        """
        last_start_date = None
        for platform_start_date in values:
            if last_start_date is None:
                last_start_date = copy.deepcopy(platform_start_date)
                assert last_start_date is not None
                last_start_date = cast(PlatformStartDate, last_start_date)
                continue

            if last_start_date.start_date >= platform_start_date.start_date:
                raise ValueError(
                    f"Platform start dates are not sequentially increasing:"
                    f" {platform_start_date.platform_id} with start date {platform_start_date.start_date}"
                    " is given after"
                    f" {last_start_date.platform_id} with start date {last_start_date.start_date}."
                )

            last_start_date = copy.deepcopy(platform_start_date)

        return values

    def platform_for_id(self, platform_id: SUPPORTED_PLATFORM_ID) -> Platform:
        """Return the Platform for the given platform ID."""
        return platform_for_id(platforms=self.platforms, platform_id=platform_id)

    def get_platform_by_date(
        self,
        date: dt.date,
    ) -> Platform:
        """Return the platform for this date."""
        first_start_date = self.get_first_platform_start_date()
        if date < first_start_date:
            raise RuntimeError(
                f"""
            date {date} too early.
            First start_date: {first_start_date}
            """
            )

        return_platform_id = None
        for cdr_platform_start_date in self.cdr_platform_start_dates:
            if date >= cdr_platform_start_date.start_date:
                return_platform_id = cdr_platform_start_date.platform_id
                continue
            else:
                break

        if return_platform_id is None:
            raise RuntimeError(f"Could not find platform for {date=}")

        return self.platform_for_id(return_platform_id)

    def get_first_platform_start_date(self) -> dt.date:
        """Return the start date of the first platform."""
        earliest_date = self.cdr_platform_start_dates[0].start_date

        return earliest_date

    def platform_available_for_date(
        self, platform_id: SUPPORTED_PLATFORM_ID, date: dt.date
    ) -> bool:
        """Indicates if the given platform ID is avaiable for the provided date."""
        platform = self.platform_for_id(platform_id)
        if date < platform.date_range.first_date:
            return False

        last_date = platform.date_range.last_date
        if last_date is not None and date > last_date:
            return False

        return True
