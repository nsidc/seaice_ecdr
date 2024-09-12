#!/bin/bash

set -exo pipefail

BASE_OUTPUT_DIR="/share/apps/G02202_V5/25km/combined/"

./scripts/cli.sh monthly --year 2022 --month 3 --hemisphere north --base-output-dir $BASE_OUTPUT_DIR --resolution 25 --ancillary-source CDRv4
