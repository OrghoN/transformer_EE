#!/bin/bash

SCRIPT_DIR_M="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running $SCRIPT_DIR_M/"

source $SCRIPT_DIR_M/0/test.bash
source $SCRIPT_DIR_M/1/test.bash
source $SCRIPT_DIR_M/2/test.bash
source $SCRIPT_DIR_M/3/test.bash
source $SCRIPT_DIR_M/4/test.bash
source $SCRIPT_DIR_M/5/test.bash


