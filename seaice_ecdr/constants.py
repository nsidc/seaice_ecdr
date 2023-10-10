from pathlib import Path

# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path("/share/apps/amsr2-cdr")

# Location of LANCE AMSR2 NRT data files:
LANCE_NRT_DATA_DIR = NSIDC_NFS_SHARE_DIR / "lance_amsr2_nrt_data"
