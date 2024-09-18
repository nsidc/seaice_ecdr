#!/bin/bash

set -exo pipefail

BASE_OUTPUT_DIR="/share/apps/G02202_V5/25km/combined/"
mkdir -p $BASE_OUTPUT_DIR

START_DATE="2022-03-01"
END_DATE="2022-04-01"
HEMISPHERE="north"

./scripts/cli.sh daily --date $START_DATE --end-date $END_DATE --hemisphere $HEMISPHERE --base-output-dir $BASE_OUTPUT_DIR
