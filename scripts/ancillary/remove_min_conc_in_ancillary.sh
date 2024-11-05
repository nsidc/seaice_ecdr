#!/bin/bash

# ./remove_min_conc_in_ancillary.sh

# if there is a field called "min_concentration" in arg1, drop it

# Note: This command outputs a file with the same basename
#       in the current directory.
#       And it does not allow overwriting of the original file

# Sample usage:
# This script was used to remove the min_concentration field
#   from the v05r00 ancillary files:
#   ./remove_min_conc_in_ancillary.sh /share/apps/G02202_V5/v05_ancillary/G02202-ancillary-psn25-v05r00.nc 
#   ./remove_min_conc_in_ancillary.sh /share/apps/G02202_V5/v05_ancillary/G02202-ancillary-pss25-v05r00.nc 


var_to_delete=min_concentration

ifn="$1"
if [ -z "$ifn" ]; then
  echo "No input file given.  (Should be a netCDF file)"
  exit
fi

ofn=$(basename ${ifn})

if [ "$ifn" == "$ofn" ]; then
  echo "input and output filename would be the same, exiting"
  echo "  ${ofn}"
  exit
fi

if [ "$ifn" == ./"$ofn" ]; then
  echo "input and output filename would be the same, exiting"
  echo "  ${ofn}"
  exit
fi

cmd_output=$(ncks -C -O -h -x -v ${var_to_delete} ${ifn} ${ofn})
if [ ! -z ${cmd_output} ]; then
  echo "Command output not empty.  Perhaps something went wrong?"
  echo "${cmd_output}"
fi
