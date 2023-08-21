import pandas as pd
import os
from tqdm import tqdm

class DataframeCreator:
    '''
    This class creates a dataframe from the files in a directory.
    The dataframe will have the following columns:

    Attributes:
    -----------
    input_path: string
        path to the directory where the files are stored.
    output_path: string
        path to the directory where the dataframe will be stored.
    df_name: string
        name of the dataframe.

    Methods:
    --------
    species2int(species)
        Returns the int that corresponds to the entered species
    create_df()
        Creates the dataframe and stores it in the output_path directory.
    get_df_path()
        Returns the path to the dataframe.
    '''

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df_name = 'df_' + self.input_path.split('/')[-1] + '.pkl'


    def species2int(self, species):
        '''
        Returns the int that corresponds to the entered species:
        Blue: 0
        Fin: 1
        Unidentified: 2
        Minke: 3
        Humpback: 4
        
        Arguments
        ---------
        species: string
            species in string format
        '''
        if species == 'Blue':
            return 0
        elif species == 'Fin':
            return 1
        elif species == 'Unidentified':
            return 2
        elif species == 'Minke':
            return 3
        elif species == 'Humpback':
            return 4
        else:
            return 5
    
    @staticmethod
    def check_filename(filename):
        # Extract the base filename without extension
        base_filename = os.path.splitext(filename)[0]

        # Check if the last two characters of the base filename are "SS"
        if base_filename[-2:] == "SS":
            return True
        else:
            return False
    
    def create_df(self):

        cols = ['subdataset', 'wav_name','species', 'num_species', 'vocalization', 'date', 'begin_sample', 'end_sample', 'sampling_rate', 'SS']
        df = pd.DataFrame(columns=cols)

        print(f'Creating {self.df_name}')

        for f in tqdm(os.listdir(self.input_path)):
            subdirectory = f.split('_')[0]
            wav_name = f.split('_')[1]
            species = f.split('_')[2]
            vocalization = f.split('_')[3]
            date = f.split('_')[4]
            begin_sample = f.split('_')[5]
            end_sample = f.split('_')[6]
            sample_rate = f.split('_')[7].split('H')[0]
            SS = DataframeCreator.check_filename(f)
            
            row = [subdirectory, wav_name, species, self.species2int(species), vocalization, date, begin_sample, end_sample, sample_rate, SS]
            df.loc[len(df)] = row
        
        df_path = os.path.join(self.output_path, self.df_name)
        df.to_pickle(df_path)
        return df_path
    
    def get_df_path(self):

        df_path = os.path.join(self.output_path, self.df_name)
        if os.path.exists(df_path) and os.path.isfile(df_path):
            return df_path
        else:
            return self.create_df()

