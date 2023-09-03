# SPECTRAL SUBTRACTION

To implement Spectral Subtraction we will need to create a uniform DATASET with all the samples. 

1. The first step is to create the audios with spectral substraction. To do so, we will checkout to ```feature/spectral_subtraction``` branch and run the Jaume's code:

This is an example with the $\alpha=10$:

```
srun -A veu -p veu --mem=16G -c 8  python src/apply_spectral_subtraction.py --spectral_subtraction_prob 1.0 '/home/usuaris/veussd/DATABASES/Ocean/Cleaned_AcousticTrends_min5/data' '/home/usuaris/veussd/DATABASES/Ocean/SPECTRAL_SUBTRACTION/SS_10'
```
This is done with $\alpha\in\{0.1, 1, 5, 10, 50\}$.
