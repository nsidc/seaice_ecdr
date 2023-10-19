#!/bin/bash

ARGS=$@

THIS_DIR="$( cd "$(dirname "$0")"; pwd -P )"
PM_ICECON_DIR=/home/vagrant/pm_icecon
PM_TB_DATA_DIR=/home/vagrant/pm_tb_data

# PYTHONPATH=$THIS_DIR/.. python -m pm_icecon.cli.entrypoint $ARGS
# PYTHONPATH=$THIS_DIR/..:${PM_ICECON_DIR} python -m seaice_ecdr.cli.entrypoint $ARGS
PYTHONPATH=$THIS_DIR/..:${PM_ICECON_DIR}:${PM_TB_DATA_DIR} python -m pm_icecon.cli.entrypoint $ARGS
