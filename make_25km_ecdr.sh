#!/bin/bash

set -exo pipefail

BT_NT_BASE_DIR="/share/apps/G02202_V5/25km/BT_NT"
mkdir -p $BT_NT_BASE_DIR
NT2_BASE_DIR="/share/apps/G02202_V5/25km/NT2"
mkdir -p $NT2_BASE_DIR

START_DATE="2020-01-01"
END_DATE="2021-01-01"
HEMISPHERE="north"

# NT method first
FORCE_PLATFORM=F17 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BT_NT_BASE_DIR --land-spillover-alg BT_NT --resolution 25
FORCE_PLATFORM=am2 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BT_NT_BASE_DIR --land-spillover-alg BT_NT --resolution 25

# NT2 method second
FORCE_PLATFORM=F17 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $NT2_BASE_DIR --land-spillover-alg NT2 --resolution 25
FORCE_PLATFORM=am2 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $NT2_BASE_DIR --land-spillover-alg NT2 --resolution 25
