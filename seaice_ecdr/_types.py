from typing import Literal

# In kilometers.
ECDR_SUPPORTED_RESOLUTIONS = Literal["12.5", "25"]

# Supported sats
SUPPORTED_SAT = Literal[
    "am2",  # AMSR2
    "ame",  # AMSRE
    "F17",  # SSMIS F17
    "F13",  # SSMI F13
    "F11",  # SSMI F11
    "F08",  # SSMI F08
    "n07",  # Nimus SMMR
]
