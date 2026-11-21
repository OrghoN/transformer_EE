#!/bin/sh
submit_cafana.py -n 500 --print_jobsub \
--rel development -o /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/nominal \
--user_tarball ../../../transformerEE_data_extract.tar.bz2 \
/pnfs/nova/scratch/users/$USER/nova_data_extractor/mag_moment/nd_bg_extractor.C
