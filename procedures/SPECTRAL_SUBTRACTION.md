# SPECTRAL SUBTRACTION

To implement Spectral Subtraction we will need to create a uniform DATASET with all the samples. 
0. The initial step in order to start even thinking in spectral subtraction is to have an estimation of the noise. Spectral subtraction is an easy concept algorithm, but this step can be challenging.

For each audio file we want its specific noise estimation. Because we are working with whale audios, we can assume that the vocalizations are very isolated. For this reason, we can consider the previous seconds of the vocalization as the estimation. Formally, if we consider $s$ and $e$ the initial and end of the vocalization respectively, we will define the noise of this audio as 

$$[s-10, e-10]$$

So we will need to construct an alternative dataset that will contain all the corresponding audio samples of noise that match each of the ones of the original dataset.

To do so, we will run the following code:

```
python3 src/extraction.py "raw_dataset" "final_dataset" --min_frame_size_sec 5
```

1. The first step is to create the audio with spectral subtraction. To do so, we will checkout to ```feature/spectral_subtraction``` branch and run Jaume's code:

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

The last step is to split our dataset into TRAIN, VALID and TEST. For doing so we will make use of the ```split.py``` file. 

This is an example of how the path has been configurated:

```python
source_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/23_09_04_12_07_04_jmexpjiq_icy-valley-34'
train_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/TRAIN'
valid_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/VALID'
test_directory = '/home/usuaris/veussd/DATABASES/Ocean/SPECTROGRAMS_SS/SS_50/TEST'
```
