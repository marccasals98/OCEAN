# SPECTRAL SUBTRACTION

To implement Spectral Subtraction we will need to create a uniform DATASET with all the samples. 

1. The first step is to create the audios with spectral substraction. To do so, we will checkout to ```feature/spectral_subtraction``` branch and run the Jaume's code:

This is an example with the $\alpha=10$:

```
srun -A veu -p veu --mem=16G -c 8  python src/apply_spectral_subtraction.py --spectral_subtraction_prob 1.0 '/home/usuaris/veussd/DATABASES/Ocean/Cleaned_AcousticTrends_min5/data' '/home/usuaris/veussd/DATABASES/Ocean/SPECTRAL_SUBTRACTION/SS_10'
```
This is done with $\alpha\in\{0.1, 1, 5, 10, 50\}$.

The Spectrograms will be saved in the following directory:
```/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS```

2. Spectrogram generation:

We will generate the spectrograms in the folder: 

```/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS```

3. Split.

The last step is to split our dataset into TRAIN, VALID and TEST. For doing so we will make usage of the ```split.py``` file. 

This is an example of how the path has been configurated:

```python
source_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/23_09_04_12_07_04_jmexpjiq_icy-valley-34'
train_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/TRAIN'
valid_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/VALID'
test_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/TEST'
```