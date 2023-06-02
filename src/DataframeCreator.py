import pandas as pd
import os
from tqdm import tqdm

class DataframeCreator:

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df_name = 'df_' + self.input_path.split('/')[-1] + '.pkl'


    def species2int(self, species):
        '''
        Retutrns the int that corresponds to the entered species:
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
        
    def create_df(self):

        cols = ['subdataset', 'wav_name','species', 'num_species', 'vocalization', 'date', 'begin_sample', 'end_sample', 'sampling_rate']
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
            row = [subdirectory, wav_name, species, self.species2int(species), vocalization, date, begin_sample, end_sample, sample_rate]
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

