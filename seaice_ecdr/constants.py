from pathlib import Path

# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path("/share/apps/amsr2-cdr")

# Outputs from the `seaice_ecdr` go to these locations.
BASE_OUTPUT_DIR = NSIDC_NFS_SHARE_DIR / "ecdr_outputs"
# Daily initial output from 'final' input data (e.g., AU_SI12)
INITIAL_DAILY_OUTPUT_DIR = BASE_OUTPUT_DIR / "final" / "initial_daily"
# Daily initial output from NRT LANCE AMSR2 data (AU_SI12_NRT_R04)
NRT_INITIAL_DAILY_OUTPUT_DIR = BASE_OUTPUT_DIR / "nrt" / "initial_daily"

# Location of LANCE AMSR2 NRT data files:
# TODO: nest the subdir under an `ecdr_inputs` or similar?
LANCE_NRT_DATA_DIR = NSIDC_NFS_SHARE_DIR / "lance_amsr2_nrt_data"

CDR_DATA_DIR = NSIDC_NFS_SHARE_DIR / "cdr_data"
