#!/bin/bash

set -exo pipefail

BT_NT_BASE_DIR="/share/apps/G02202_V5/25km/BT_NT"
mkdir -p $BT_NT_BASE_DIR
NT2_BASE_DIR="/share/apps/G02202_V5/25km/NT2"
mkdir -p $NT2_BASE_DIR

START_DATE="2012-07-03"
END_DATE="2023-12-31"

for HEMISPHERE in north south; do
    echo "Proeccesing $HEMISPHERE"
    # SSMIS/F17: BT+NT land spillover
    FORCE_PLATFORM=F17 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BT_NT_BASE_DIR --land-spillover-alg BT_NT --resolution 25


    # AMSR2: NT2
    FORCE_PLATFORM=am2 ./scripts/cli.sh multiprocess-daily --start-date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $NT2_BASE_DIR --land-spillover-alg NT2 --resolution 25
done
