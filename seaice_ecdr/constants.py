from pathlib import Path

# This is the version string for the ECDR product.
ECDR_PRODUCT_VERSION = "v05r01"

# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path("/share/apps/G02202_V5")

# Logs from running the ECDR code are saved here.
LOGS_DIR = NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_logs"

# TODO: dev-specific directories for the outputs!

# Outputs from the `seaice_ecdr` go to these locations.
BASE_OUTPUT_DIR = NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_outputs"

# Daily initial/intermediate output for 'standard' (not NRT) ECDR processing
# (e.g, using input data from AU_SI12)
STANDARD_BASE_OUTPUT_DIR = BASE_OUTPUT_DIR / "standard"

# Daily initial/intermediate output for 'Near Real Time' (NRT) ECDR processing
# (using data from AU_SI12_NRT_R04)
NRT_BASE_OUTPUT_DIR = BASE_OUTPUT_DIR / "nrt"

# Location of LANCE AMSR2 NRT data files:
# TODO: nest the subdir under an `ecdr_inputs` or similar?
LANCE_NRT_DATA_DIR = NSIDC_NFS_SHARE_DIR / "lance_amsr2_nrt_data"

# Location of surface mask & geo-information files.
CDR_ANCILLARY_DIR = NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_ancillary"
