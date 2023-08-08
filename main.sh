#!/bin/bash
#SBATCH -A veu
#SBATCH -p veu
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:6

python src/main.py > /home/usuaris/veu/pol.cavero/OCEAN/src/log_700_80_10_10.log 2>&1
