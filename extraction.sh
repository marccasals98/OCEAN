#!/bin/bash
srun -A veu -p veu --mem=16G -c 8  python src/extraction.py \
--min_frame_size_sec 5 \
#--spectral_subtraction True \
'/home/usuaris/veussd/DATABASES/Ocean/AcousticTrends_BlueFinLibrary' \
'/home/usuaris/veussd/DATABASES/Ocean/SS_Cleaned_AcousticTrends_16'
