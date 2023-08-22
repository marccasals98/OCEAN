# OCEAN
## Operator-Centered Enhancement of Awareness in Navigation
<img src="https://github.com/marccasals98/OCEAN/blob/main/OCEAN.png" width="300">

## How to run:

1. Create a Conda environment. For this you need to have Conda or Miniconda installed.

```bash
conda create --name name_env python=3.10
```

2. Activate the environment.

```bash
conda activate name_env
```

3. Install the dependencies.

```bash
pip install -r requirements.txt
```

4. Train.
```bash
python src/main.py
```
4b. Alternatively, to train in CALCULA (UPC)

```
srun -A veu -p veu --mem=16G -c 2 --gres=gpu:4 python src/main.py
```
To use this code, it is needed the ```.wav``` files in three different folders: ```train``` ```valid``` and ```test```. The code will automatically create a pandas dataframe for every dataset. If it is needed to change the data once we have already trained, for example for using data with spectral subtraction, it will be required to delete the previous dataframes. 
### Enhancing operator awareness in navigation
The OCEAN project is focused on enhancing operator awareness in navigation, to reduce the frequency of severe accidents like collision and grounding, to mitigate ship-strike risks to marine mammals, and to mitigate the risk presented by floating obstacles to ships.

### Accident root causes
The OCEAN project will contribute to an improved understanding of accident root causes, and will strive to reduce the resulting human, environmental and economic losses through socio-technical innovations supporting ship navigators.

### Western Norway University of Applied Sciences
The OCEAN consortium, coordinated by Western Norway University of Applied Sciences, includes 13 partner organizations across 7 different European countries from the industry, academia, NGOs and end users

### Award
The project was awarded funding by Horizon Europe and launched in October 2022, and it is due to run until 2025. 
