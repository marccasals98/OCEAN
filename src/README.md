# README

In this folder, you will find the files used to develop the different models.

The ```main.py``` is responsible for running the training. The standard way of using Pytorch is to create this file in conjunction with the ```dataset.py``` and ```model.py```.
This first file is responsible for creating the Dataset class that will load the sample and the label for each index. The other file, ```{model}.py``` will create the model. In this repository
we load the ```{model}``` as the different names that the model can take, such as *LeNet* or *ResNet*...

There are other additional files that take action in more personalized tasks:

* ```DataframeCreator.py```: Is responsible for creating the Pandas Dataframe that will contain all the features related to each audio.
* ```explorer.py```: This is an auxiliary file that is created to see how are the spectrograms created. It actually takes some samples.
* ```extraction.py```: ask Jaume.
* ```feature_extractor_Pol.py```: This file transforms the different audios in different spectrograms.
* ```metrics.py```: This file contains different metrics that use the library torchmetrics. It also contains a function to visualize the confusion matrix made by scikitlearn.
* ```split.py```: This script is run to make the data partition.
* ```stats.py```: This file is responsible to take some statistics of the dataset.

## Git procedure

The branch ```main``` is the latest stable version of the project. The branch ```develop``` is the previous step, which tries to encompass all changes made independently from all branches.
Lately, the branches that are made to make substantial changes to the main code are named ```type/description```.

Where ```type``` can take the following values:
1. ```feature```: A new change that we are adding.
2. ```hotfix```: A revision of something that we did wrong.
3. ```bug```: A bug in the code that we may encounter.

An example of this type of working could be creating a branch named ```feature/data_augmentation```where we would do the data augmentation. 

## Spectral Subtraction

The code to run spectral subtraction is:

```
srun -A veu -p veu --mem=16G -c 8  python src/apply_spectral_subtraction.py --spectral_subtraction_prob 1.0 'input_path/data/' 'output_path'
```

