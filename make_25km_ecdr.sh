#!/bin/bash

set -exo pipefail

BASE_OUTPUT_DIR="/share/apps/G02202_V5/25km/combined/"
mkdir -p $BASE_OUTPUT_DIR

START_DATE="2022-03-01"
END_DATE="2022-04-01"
HEMISPHERE="north"

# Use the default platform start dates, which excludes AMSR2
./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BASE_OUTPUT_DIR --land-spillover-alg BT_NT --resolution 25 --ancillary-source CDRv4

# Force the use of AMSR2
PLATFORM_START_DATES_CONFIG_FILEPATH=seaice_ecdr/config/prototype_platform_start_dates.yml ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BASE_OUTPUT_DIR --land-spillover-alg BT_NT --resolution 25 --ancillary-source CDRv4

# Combine where there's overlap between the two, and publish to an output dir
# that's ready for publication.

# TODO:
# * Change how checksum files are generated. They won't be "complete" anymore,
#   they'll be in "ready_for_publication" or something similar (could change
#   current `complete` to a subdir of `intermediate`). Should probably start by changing the output dir for complete daily to an intermediate dir.
# * Change when/how aggregrate files are generated. They should include the
#   prototype fields.
# * validation/qa script needs to be updated to support new outputs. How do we
#   validate prototype fields?
# * 
