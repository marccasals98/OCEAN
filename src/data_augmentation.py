import sys
# adding DA_programV2 to the system path
sys.path.insert(0, '/home/usuaris/veu/jaume.prats.cristia/workspace/ocean/code')

import os
from tqdm import tqdm
from DA_programV2.src import functions

def main():
    input_dir = '/home/usuaris/veussd/DATABASES/Ocean/Cleaned_AcousticTrends_min5'
    output_dir = '/home/usuaris/veussd/DATABASES/Ocean/test_DA_min5'

    for f in tqdm(os.listdir(input_dir)):
        input_file = os.path.join(input_dir,f)
        output_file = os.path.join(output_dir,f)
        functions.clipping(input_file, output_file)

if __name__=="__main__":
    main()