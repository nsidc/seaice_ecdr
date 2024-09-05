from typing import Literal

# In kilometers.
ECDR_SUPPORTED_RESOLUTIONS = Literal["12.5", "25"]

# Supported sats
# TODO: ideally this is used sparingly. The code should accept any number of
# platform configurations, and those configurations should defined what's
# "supported". We could even include the import string for a fetch function that
# conforms to a spec for each platform, so that the e.g., `tb_data` module does
# not need to map specific IDs to functions. See:
# https://docs.pydantic.dev/2.3/usage/types/string_types/#importstring
SUPPORTED_PLATFORM_ID = Literal[
    "am2",  # AMSR2
    "ame",  # AMSRE
    "F17",  # SSMIS F17
    "F13",  # SSMI F13
    "F11",  # SSMI F11
    "F08",  # SSMI F08
    "n07",  # Nimbus-7 SMMR
]
