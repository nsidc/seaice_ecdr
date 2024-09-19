"""Constants for ecdr processing

TODO: paths are specific to NSIDC NFS infrastructure. Eventually, it would be
nice to Dockerize this project and expose necessary directories through
volumes. E.g., the program could always write outputs to `/outputs/` and it
would be up to the runner to map `/outputs/` to a location on local disk. For
NSIDC infra, that might be `/share/apps/G02202_v5/production/` and on someone's
laptop it might be `/local-storage/foo/bar/outputs/`.
"""

import copy
import getpass
import os
import subprocess
from pathlib import Path

# This is the version string for the ECDR product.
ECDR_PRODUCT_VERSION = "v05r00"

# NSIDC infrastructure-specific paths:
NSIDC_NFS_SHARE_DIR = Path("/share/apps/G02202_V5")
if not NSIDC_NFS_SHARE_DIR.is_dir():
    raise RuntimeError(f"Expected {NSIDC_NFS_SHARE_DIR} to exist, but it does not.")

# Logs from running the ECDR code are saved here.
LOGS_DIR = NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_logs"


# environment-specific directories for outputs
def _get_env_subdir_str() -> str:
    # Get the environment, defaulting to "dev".
    environment = os.environ.get("ENVIRONMENT", "dev")

    subdir_str = copy.copy(environment)

    # in dev env, get the user.
    if environment == "dev":
        # On NSIDC development VMs, this will give the username of the user who
        # provisioned the VM.
        result = subprocess.run(
            "facter provisioned_by",
            shell=True,
            capture_output=True,
        )
        # If the above doesn't work, then we might not be on an NSIDC VM (e.g.,
        # a laptop) and we should just use the login username.
        if result.returncode != 0:
            user = getpass.getuser()
        # If the above did work, (returncode=0), then the username can be read
        # from stdout.
        else:
            user = result.stdout.decode("utf8").strip()

        subdir_str += f"/{user}"

    return subdir_str


# Outputs from the `seaice_ecdr` go to these locations by default. The CLI
# provides the option to change this.
DEFAULT_BASE_OUTPUT_DIR = (
    NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_outputs" / _get_env_subdir_str()
)
DEFAULT_BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Location of LANCE AMSR2 NRT data files:
# TODO: nest the subdir under an `ecdr_inputs` or similar?
LANCE_NRT_DATA_DIR = NSIDC_NFS_SHARE_DIR / "lance_amsr2_nrt_data"

# Location of surface mask & geo-information files.
CDR_ANCILLARY_DIR = NSIDC_NFS_SHARE_DIR / f"{ECDR_PRODUCT_VERSION}_ancillary"
CDRv4_ANCILLARY_DIR = NSIDC_NFS_SHARE_DIR / "cdrv4_equiv_ancillary"
