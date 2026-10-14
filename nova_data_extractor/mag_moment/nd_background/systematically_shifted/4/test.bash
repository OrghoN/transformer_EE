#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running $SCRIPT_DIR/mprod6_exporter_transformer_ee_nd_fhc_nonswap.C"

cafe -bq -l 5 $SCRIPT_DIR/mprod6_exporter_transformer_ee_nd_fhc_nonswap.C
