import os
import numpy as np
import pandas as pd
from  scipy.io import wavfile 
from tqdm import tqdm
from datetime import datetime
import argparse

class Extractor:

    def __init__(self, dataset_path, output_path):

        self.dataset_path = dataset_path
        self.output_path = output_path

        # pd dataframe for extraction statistics:
        cols = ['subdataset', 'original name of wav file', 'species', 'vocalization', 'date', 'first sample from orig. file', 'last sample from orig. file', 'sampling frequency in Hz']
        self.extraction_df = pd.DataFrame(columns=cols)
    
    def scan_dataset(self):
        '''
        Scan dataset directory to create list of subdataset directories.

        Arguments
        ---------

        Returns
        -------
        subdataset_dirs: list
            list of subdataset directories found.
        '''
        subdatasets_dirs = []
        for f in os.listdir(self.dataset_path):
            path = os.path.join(self.dataset_path, f)
            if os.path.isdir(path) and not f[0].isdigit(): # to avoid saving documentation and other dir that start with a digit
                subdatasets_dirs.append(f)
        subdatasets_dirs.reverse()
        print('Found dataset subdirectories:')
        print(subdatasets_dirs)
        return subdatasets_dirs
    
    def extract_species(self, annotation_file_name):
        '''
        Extract species name and vocalization from the annotation file.

        Arguments
        ---------
        annotation_file_name : string
            file name of the annotations

        Returns
        -------
        species: string
            name of the species
        vocalization: string
            code of the vocalization
        '''

        annotation_file_name = annotation_file_name.lower().replace('.','').replace('_','').replace('-','') # lower case and delete '.' and '_' to cover the annotation names across datasets

        substrings_fin20plus = ['Bp20Plus'.lower(), 'Fin20Plus'.lower()]            # possible namings of Fin 20Plus call
        substrings_fin20hz = ['Bp20Hz'.lower(), 'Fin20Hz'.lower(), 'Bp20'.lower()]  # possible namings of Fin 20Hz call
        substrings_findwnswp = ['BpD'.lower(), 'FinD'.lower()]                      # possible namings of Fin Downsweep call

        if 'BmAntA'.lower() in annotation_file_name:
            species = 'Blue'
            vocalization = 'Bm-Ant-A'
        elif 'BmAntB'.lower() in annotation_file_name:
            species = 'Blue'
            vocalization = 'Bm-Ant-B'
        elif 'BmAntZ'.lower() in annotation_file_name:
            species = 'Blue'
            vocalization = 'Bm-Ant-Z'
        elif 'BmD'.lower() in annotation_file_name:
            species = 'Blue'
            vocalization = 'Bm-D'
        elif 'BmSWI'.lower() in annotation_file_name:
            species = 'Blue'
            vocalization = 'Bm-SWI'
        elif any(substring in annotation_file_name for substring in substrings_fin20plus):
            species = 'Fin'
            vocalization = 'Bp-20Plus'
        elif any(substring in annotation_file_name for substring in substrings_fin20hz):
            species = 'Fin'
            vocalization = 'Bp-20Hz'
        elif any(substring in annotation_file_name for substring in substrings_findwnswp):
            species = 'Fin'
            vocalization = 'Bp-Downsweep'
        elif 'BpHigherCall'.lower() in annotation_file_name:
            species = 'Fin'
            vocalization = 'Bp-HigherCall'
        elif 'Unid'.lower() in annotation_file_name:
            species = 'Unidentified'
            vocalization = ''
        elif 'Minke'.lower() in annotation_file_name:
            species = 'Minke'
            vocalization = ''
        elif 'Hump'.lower() in annotation_file_name:
            species = 'Humpback'
            vocalization = ''
        else:
            raise ValueError('Not able to identify species from file name')
        
        return species, vocalization

    def extract_date(self, wav_file, subdataset):
        '''
        Extract date from the wav file name and subdataset

        Arguments
        ---------
        wav_file : string
            wav file name of the
        subdataset: string
            name of the subdataset

        Returns
        -------
        date: string
            date in format 20220511
        '''

        if subdataset.lower() == 'casey2014' or subdataset.lower() == 'kerguelen2014':
            date = wav_file.split('_')[1].replace('-','')
        else:
            date = wav_file.split('_')[0]
        return date

    def print_extraction_report(self, log_file, n_extracted, n_not_extracted, n_species_not_identified, n_file_not_found, n_multifile_event):
        '''
        Prints extraction report in console and in log_file

        Arguments
        ---------
        log_file: file
        n_extracted: int
        n_not_extracted: int
        n_species_not_identified: int
        n_file_not_found: int 
        n_multifile_event: int

        '''

        # printing report in console:
        print('\n----------------------------------------------------------')
        print('EXTRACTION COMPLETED:')
        print(f'Total events extracted: {n_extracted}')
        print(f'Total events not extracted: {n_not_extracted}')
        print(f'    due to unidentified species: {n_species_not_identified}')
        print(f'    due to not found file: {n_file_not_found}')
        print(f'    due to starting and ending in different files: {n_multifile_event}')
        print('----------------------------------------------------------')

        # printing report in log file:
        log_file.write('----------------------------------------------------------\n')
        log_file.write('EXTRACTION COMPLETED:\n')
        log_file.write(f'Total events extracted: {n_extracted}\n')
        log_file.write(f'Total events not extracted: {n_not_extracted}\n')
        log_file.write(f'    due to unidentified species: {n_species_not_identified}\n')
        log_file.write(f'    due to not found file: {n_file_not_found}\n')
        log_file.write(f'    due to starting and ending in different files: {n_multifile_event}\n')
        log_file.write('----------------------------------------------------------\n')
    
    def extract(self, subdatasets_dirs):
        '''
        Extract annotated events into specified output path.

        Arguments
        ---------
        subdatasets_dirs: list
            list of the subdatasets to extract.

        '''
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path) # create output dir in case it doesnt exists.
        
        # create log file with date and time of execution
        now = datetime.now()
        date_time_filename = now.strftime("%Y%m%d-%H.%M.%S")
        date_time_txt = now.strftime("%Y-%m-%d at %H:%M:%S")
        log_file = open(os.path.join(self.output_path, 'log_file_' + date_time_filename + '.txt'), 'w')
        log_file.write(f'Log file for extraction executed on {date_time_txt}:\n')
        log_file.write('--------------------------------------------------------------------\n')
        log_file.write('Naming criterion for extracted wav files:\n')
        log_file.write('{subdataset}_{original name of wav file}_{species}_{vocalization}_{date}_{first sample from orig. file}_{last sample from orig. file}_{sampling frequency in Hz}.wav\n')
        log_file.write('--------------------------------------------------------------------\n')
        log_file.write('\n LOG:\n')

        # initialize counters for extraction report
        not_extracted_counter = 0
        file_not_found_counter = 0
        species_not_identified_counter = 0
        multifile_event_counter = 0
        extracted_counter = 0

        for i, subdirectory in enumerate(subdatasets_dirs): # iterate through subdataset directories
            print(f'Extracting {subdirectory} ...')
            log_file.write(f'Extracting {subdirectory} ...\n')
            annotations_subdir = os.path.join(self.dataset_path, subdirectory)
            for filename in os.listdir(annotations_subdir): # iterate through annotations in subdataset directory
                if filename.endswith('.txt'): # check that it is an annotation file (.txt)
                    ann = pd.read_csv(os.path.join(annotations_subdir, filename), sep="\t") # open annotation file as pd.dataframe
                    for index, row in tqdm(ann.iterrows(), total=ann.shape[0]): # iterate through events in annotation file
                        if row['Begin File'] == row['End File']: # event starts and ends in same file
                            wav_file = row['Begin File']
                            wav_path = os.path.join(self.dataset_path, subdirectory, 'wav', wav_file)
                            try: # try extracting the annotated event in wav file
                                sample_rate, sig = wavfile.read(wav_path) # open file
                                begin_sample = row['Beg File Samp (samples)']
                                end_sample = row['End File Samp (samples)']
                                sig_event = sig[begin_sample:end_sample] # extract event
                                # saving and defining output file name:
                                wav_name, extension = os.path.splitext(wav_file)
                                wav_name = wav_name.replace('_','-')
                                species, vocalization = self.extract_species(filename)
                                date = self.extract_date(wav_file, subdirectory)
                                output_file_name = subdirectory + "_" + wav_name + "_" + species + "_" + vocalization + "_" + date + "_" + str(begin_sample) + "_" + str(end_sample) + "_" + str(sample_rate) + "Hz.wav"
                                wavfile.write(os.path.join(self.output_path, output_file_name), sample_rate, sig_event)
                                # add data to the extraction dataframe:
                                row = [subdirectory, wav_name, species, vocalization, date, begin_sample, end_sample, sample_rate]
                                self.extraction_df.loc[len(self.extraction_df)] = row
                                extracted_counter += 1
                            except FileNotFoundError: # in case wav file is not found
                                not_extracted_counter += 1
                                file_not_found_counter += 1
                                d = os.path.join(self.dataset_path, subdirectory,'wav')
                                tqdm.write(f'FILE: {wav_file} NOT FOUND IN: {d}')
                                log_file.write(f'FILE: {wav_file} NOT FOUND IN: {d}\n')
                            except ValueError: # in case not able to extract species from annotation file
                                not_extracted_counter += ann.shape[0]
                                species_not_identified_counter += ann.shape[0]
                                tqdm.write(f'Not able to extract species and vocalization from: {filename}')
                                log_file.write(f'Not able to extract species and vocalization from: {filename}\n')
                                break

                        else: # event starts and ends in different files
                            not_extracted_counter += 1
                            multifile_event_counter += 1
                            tqdm.write('Event not extracted: Starts and ends in different files')
                            log_file.write('Event not extracted: Starts and ends in different files\n')
                            
        self.extraction_df.to_pickle(os.path.join(self.output_path, 'extraction_df.pkl'))
        self.print_extraction_report(log_file, extracted_counter, not_extracted_counter, species_not_identified_counter, file_not_found_counter, multifile_event_counter)
        log_file.close()

def main(dataset_path, output_path):
    extractor = Extractor(dataset_path, output_path)
    subdatasets = extractor.scan_dataset()
    extractor.extract(subdatasets)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extract events from the specified dataset')

    parser.add_argument('--dataset_path', type=str, help='path of the dataset to extract')
    parser.add_argument('--output_path', type=str, help='directory where extracted events and log_file are saved')

    params=parser.parse_args()

    main(params.dataset_path, params.output_path)