#!/bin/bash
#SBATCH -o logs/sbatch/outputs/slurm-%j.out
#SBATCH -e logs/sbatch/errors/slurm-%j.err
#SBATCH -p veu							# Partition to submit to
#SBATCH -c4
#SBATCH --mem=64G      					# Max CPU Memory
#SBATCH --gres=gpu:1
#SBATCH --job-name=feature_extractor_voxceleb_1_test
python src/feature_extractor.py \
    --audio_paths_file_folder './src/feature_extractor/' \
	--audio_paths_file_name 'spectrograms_5_sec_new.lst' \
	--prepend_directory '/home/usuaris/veussd/DATABASES/Ocean/NEW_SS/SS_1' \
	--dump_folder_name '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_NEW_SS/SS_1' \
	--sampling_rate 250 \
	--n_fft_secs 0.512 \
	--win_length_secs 0.512 \
	--hop_length_secs 0.128 \
	--n_mels 40 \
    > logs/voxceleb_1_test.log 2>&1