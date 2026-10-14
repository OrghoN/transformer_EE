#!/bin/sh

echo "Creating directories in scratch area for depositing the csv outputs:"

# Overall data
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data

# Dataset 
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data

# nominal
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/nominal
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/nominal
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/nominal

# systematically_shifted
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted

# Each systematically_shifted sample

# enhanced_nu-on-e_data
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/0
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/1
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/2
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/3
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/4
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/enhanced_nu-on-e_data/systematically_shifted/5

# nu_e_cc_data

mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/0
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/1
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/2
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/3
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/4
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/5
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/6
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/7
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nu_e_cc_data/systematically_shifted/8

# nd_background_data

mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/0
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/1
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/2
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/3
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/4
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/5
mkdir -p /pnfs/nova/scratch/users/$USER/transformer_ee_csv/mag_moment_data/nd_background_data/systematically_shifted/6

# Set group permission so grid can write
chmod g+w /pnfs/nova/scratch/users/$USER/transformer_ee_csv

