#!/bin/bash
#SBATCH -p veu							# Partition to submit to
#SBATCH -c4
#SBATCH --mem=64G      					# Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=dmha_ocean
python src/main.py \
    > src/log.log 2>&1