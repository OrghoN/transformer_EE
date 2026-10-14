#!/bin/bash

SCRIPT_DIR_T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running $SCRIPT_DIR_T/"

source $SCRIPT_DIR_T/enhanced_nu-on-e/nominal/test.bash
source $SCRIPT_DIR_T/enhanced_nu-on-e/systematically_shifted/test.bash

source $SCRIPT_DIR_T/nu_e_cc/nominal/test.bash
source $SCRIPT_DIR_T/nu_e_cc/systematically_shifted/test.bash

source $SCRIPT_DIR_T/nd_background/nominal/test.bash
source $SCRIPT_DIR_T/nd_background/systematically_shifted/test.bash

