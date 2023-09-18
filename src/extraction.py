import os
import numpy as np
import pandas as pd
from  scipy.io import wavfile 
from tqdm import tqdm
from datetime import datetime
import argparse
import random
from tqdm import tqdm

class NonPositiveDurationException(Exception): 
    def __init__(self, message):
        # Call the base class constructor with the custom message
        super().__init__(message)

class NonSpeciesException(Exception): 
    def __init__(self, message):
        # Call the base class constructor with the custom message
        super().__init__(message)

class Extractor:

    def __init__(self, dataset_path, output_path, min_frame_size_sec):

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.min_frame_size_sec = min_frame_size_sec

        # pd dataframe for extraction statistics:
        cols = ['subdataset', 'wav_name','species', 'num_species', 'vocalization', 'date', 'begin_sample', 'end_sample', 'sampling_rate']
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
            raise NonSpeciesException('Not able to identify species from file name')
        
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


    def print_extraction_report(self, log_file, extracted, not_extracted, species_not_identified, file_not_found, multifile_event, non_positvide_duration, not_identified_species, exec_time):
        '''
        Prints extraction report in console and in log_file

        Arguments
        ---------
        log_file: file
        extracted: list
        not_extracted: list
        species_not_identified: int
        file_not_found: list
        multifile_event: list
        non_positvide_duration: list
        not_identified_species: list
            List of the annotation files where species has not been identified.
        exec_time: datetime.timedelta
            Total execution time

        '''
        total = sum(extracted) + sum(not_extracted)
        

        # printing report in console:
        print('\n----------------------------------------------------------')
        print(f'EXTRACTION COMPLETED: {exec_time}')
        print(f"Total events extracted: {sum(extracted)} ({round(100*sum(extracted)/total, 2)}%) (Blue:{extracted[self.species2int('Blue')]}, Fin:{extracted[self.species2int('Fin')]}, Unidentified:{extracted[self.species2int('Unidentified')]}, Minke:{extracted[self.species2int('Minke')]}, Humpback:{extracted[self.species2int('Humpback')]})")
        print(f"Total events not extracted: {sum(not_extracted)} ({round(100*sum(not_extracted)/total, 2)}%) (Blue:{not_extracted[self.species2int('Blue')]}, Fin:{not_extracted[self.species2int('Fin')]}, Unidentified:{not_extracted[self.species2int('Unidentified')]}, Minke:{not_extracted[self.species2int('Minke')]}, Humpback:{not_extracted[self.species2int('Humpback')]})")
        print(f"    due to unidentified species: {species_not_identified} ({round(100*species_not_identified/total, 2)}%)")
        print(f"    due to not found file: {sum(file_not_found)} ({round(100*sum(file_not_found)/total, 2)}%) (Blue:{file_not_found[self.species2int('Blue')]}, Fin:{file_not_found[self.species2int('Fin')]}, Unidentified:{file_not_found[self.species2int('Unidentified')]}, Minke:{file_not_found[self.species2int('Minke')]}, Humpback:{file_not_found[self.species2int('Humpback')]})")
        print(f"    due to starting and ending in different files: {sum(multifile_event)} ({round(100*sum(multifile_event)/total, 2)}%) (Blue:{multifile_event[self.species2int('Blue')]}, Fin:{multifile_event[self.species2int('Fin')]}, Unidentified:{multifile_event[self.species2int('Unidentified')]}, Minke:{multifile_event[self.species2int('Minke')]}, Humpback:{multifile_event[self.species2int('Humpback')]})")
        print(f"    due to non positive duration: {sum(non_positvide_duration)} ({round(100*sum(non_positvide_duration)/total, 2)}%) (Blue:{non_positvide_duration[self.species2int('Blue')]}, Fin:{non_positvide_duration[self.species2int('Fin')]}, Unidentified:{non_positvide_duration[self.species2int('Unidentified')]}, Minke:{non_positvide_duration[self.species2int('Minke')]}, Humpback:{non_positvide_duration[self.species2int('Humpback')]})")
        if len(not_identified_species) > 0:
            print('\nUnable to identify species from:')
            for f in not_identified_species:
                print('\t' + f)
        print('----------------------------------------------------------')
        

        # printing report in log file:
        log_file.write('----------------------------------------------------------\n')
        log_file.write(f'EXTRACTION COMPLETED: {exec_time}\n')
        log_file.write(f"Total events extracted: {sum(extracted)} ({round(100*sum(extracted)/total, 2)}%) (Blue:{extracted[self.species2int('Blue')]}, Fin:{extracted[self.species2int('Fin')]}, Unidentified:{extracted[self.species2int('Unidentified')]}, Minke:{extracted[self.species2int('Minke')]}, Humpback:{extracted[self.species2int('Humpback')]})\n")
        log_file.write(f"Total events not extracted: {sum(not_extracted)} ({round(100*sum(not_extracted)/total, 2)}%) (Blue:{not_extracted[self.species2int('Blue')]}, Fin:{not_extracted[self.species2int('Fin')]}, Unidentified:{not_extracted[self.species2int('Unidentified')]}, Minke:{not_extracted[self.species2int('Minke')]}, Humpback:{not_extracted[self.species2int('Humpback')]})\n")
        log_file.write(f"\tdue to unidentified species: {species_not_identified} ({round(100*species_not_identified/total, 2)}%)\n")
        log_file.write(f"\tdue to not found file: {sum(file_not_found)} ({round(100*sum(file_not_found)/total, 2)}%) (Blue:{file_not_found[self.species2int('Blue')]}, Fin:{file_not_found[self.species2int('Fin')]}, Unidentified:{file_not_found[self.species2int('Unidentified')]}, Minke:{file_not_found[self.species2int('Minke')]}, Humpback:{file_not_found[self.species2int('Humpback')]})\n")
        log_file.write(f"\tdue to starting and ending in different files: {sum(multifile_event)} ({round(100*sum(multifile_event)/total, 2)}%) (Blue:{multifile_event[self.species2int('Blue')]}, Fin:{multifile_event[self.species2int('Fin')]}, Unidentified:{multifile_event[self.species2int('Unidentified')]}, Minke:{multifile_event[self.species2int('Minke')]}, Humpback:{multifile_event[self.species2int('Humpback')]})\n")
        log_file.write(f"\tdue to non positive duration: {sum(non_positvide_duration)} ({round(100*sum(non_positvide_duration)/total, 2)}%) (Blue:{non_positvide_duration[self.species2int('Blue')]}, Fin:{non_positvide_duration[self.species2int('Fin')]}, Unidentified:{non_positvide_duration[self.species2int('Unidentified')]}, Minke:{non_positvide_duration[self.species2int('Minke')]}, Humpback:{non_positvide_duration[self.species2int('Humpback')]})\n")
        if len(not_identified_species) > 0:
            log_file.write('\nUnable to identify species from:\n')
            for f in not_identified_species:
                log_file.write('\t' + f + '\n')
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
            os.mkdir(self.output_path) # create output dir .

        log_dir_path = os.path.join(self.output_path, 'logs')
        if not os.path.exists(log_dir_path):
            os.mkdir(log_dir_path) # create directory for saving log files

        data_dir_path = os.path.join(self.output_path, 'data')
        if not os.path.exists(data_dir_path):
            os.mkdir(data_dir_path) # create directory for saving data (wav files)
        
        
        # create log file with date and time of execution
        init_time = datetime.now()
        date_time_filename = init_time.strftime("%Y%m%d-%H.%M.%S")
        date_time_txt = init_time.strftime("%Y-%m-%d at %H:%M:%S")
        log_file = open(os.path.join(log_dir_path, 'log_file_' + date_time_filename + '.txt'), 'w')
        log_file.write(f'Log file for extraction executed on {date_time_txt}:\n')
        log_file.write('--------------------------------------------------------------------\n')
        log_file.write('Naming criterion for extracted wav files:\n')
        log_file.write('{subdataset}_{original name of wav file}_{species}_{species as a number}_{vocalization}_{date}_{first sample from orig. file}_{last sample from orig. file}_{sampling frequency in Hz}.wav\n')
        log_file.write('--------------------------------------------------------------------\n')
        log_file.write('\n LOG:\n')

        # initialize counters for extraction report
        l_extracted_counter = [0,0,0,0,0] # Counter for extracted vocalizations by species [Blue, Fin, Unidentified, Minke, Humpback]
        l_not_extracted_counter = [0,0,0,0,0]
        l_file_not_found_counter = [0,0,0,0,0]
        l_multifile_event_counter = [0,0,0,0,0]
        l_non_positvide_duration_counter = [0,0,0,0,0]
        species_not_identified_counter = 0 
        not_identified_species = []


        for i, subdirectory in tqdm(enumerate(subdatasets_dirs)): # iterate through subdataset directories
            print(f'Extracting {subdirectory} ...')
            log_file.write(f'Extracting {subdirectory} ...\n')
            annotations_subdir = os.path.join(self.dataset_path, subdirectory)
            for filename in os.listdir(annotations_subdir): # iterate through annotations in subdataset directory
                if filename.endswith('.txt'): # check that it is an annotation file (.txt)
                    ann = pd.read_csv(os.path.join(annotations_subdir, filename), sep="\t") # open annotation file as pd.dataframe
                    for index, row in tqdm(ann.iterrows(), total=ann.shape[0]): # iterate through events in annotation file
                        if row['Begin File'] == row['End File']: # event starts and ends in same file
                            if subdirectory == 'Greenwich64S2015':
                                wav_file = row['Begin File'].replace('_AWI229-11_SV1057','')
                            else:
                                wav_file = row['Begin File']
                            wav_path = os.path.join(self.dataset_path, subdirectory, 'wav', wav_file)
                            try: # try extracting the annotated event in wav file
                                sample_rate, sig = wavfile.read(wav_path) # open file
                                # saving and defining output file name:
                                wav_name, extension = os.path.splitext(wav_file)
                                wav_name = wav_name.replace('_','-')
                                species, vocalization = self.extract_species(filename)
                                num_species = self.species2int(species)
                                if row['End File Samp (samples)'] <= row['Beg File Samp (samples)']: # event with non positive duration
                                    raise NonPositiveDurationException('Event not extracted: non positive duration')
                                min_frame_size_samples = self.min_frame_size_sec * sample_rate
                                if (row['End File Samp (samples)'] - row['Beg File Samp (samples)']) < min_frame_size_samples: # event shorter than min_frame_size
                                    event_begin_sample = row['Beg File Samp (samples)']
                                    event_end_sample = row['End File Samp (samples)']
                                    begin_sample = random.randint(event_end_sample - min_frame_size_samples, event_begin_sample)
                                    end_sample = begin_sample + min_frame_size_samples
                                    if begin_sample < 0: # begin_sample falls outside the audio
                                        # correct samples to fall inside
                                        end_sample = end_sample - begin_sample
                                        begin_sample = 0
                                    elif end_sample >= len(sig): # end_sample falls outside the audio
                                        # correct samples to fall inside
                                        diff = end_sample - (len(sig)-1)
                                        end_sample = end_sample -diff
                                        begin_sample =  begin_sample - diff
                                else:
                                    begin_sample = row['Beg File Samp (samples)']
                                    end_sample = row['End File Samp (samples)']
                                sig_event = sig[begin_sample-10*sample_rate:end_sample-10*sample_rate] # extract event
                                date = self.extract_date(wav_file, subdirectory)
                                output_file_name = subdirectory + "_" + wav_name + "_" + species + "_" + vocalization + "_" + date + "_" + str(begin_sample) + "_" + str(end_sample) + "_" + str(sample_rate) + "Hz.wav"
                                wavfile.write(os.path.join(data_dir_path, output_file_name), sample_rate, sig_event)
                                # add data to the extraction dataframe:
                                row = [subdirectory, wav_name, species, num_species, vocalization, date, begin_sample, end_sample, sample_rate]
                                self.extraction_df.loc[len(self.extraction_df)] = row
                                #print(species)
                                #print(self.species2int(species))
                                l_extracted_counter[self.species2int(species)] += 1
                                #extracted_counter += 1

                            except NonPositiveDurationException as e: # event with non positive duration
                                species, vocalization = self.extract_species(filename)
                                l_not_extracted_counter[self.species2int(species)] += 1
                                l_non_positvide_duration_counter[self.species2int(species)] += 1
                                tqdm.write(f'{str(e)}')
                                log_file.write(f'{str(e)}')

                            except FileNotFoundError: # in case wav file is not found
                                #not_extracted_counter += 1
                                #file_not_found_counter += 1
                                species, vocalization = self.extract_species(filename)
                                l_not_extracted_counter[self.species2int(species)] += 1
                                l_file_not_found_counter[self.species2int(species)] += 1
                                d = os.path.join(self.dataset_path, subdirectory,'wav')
                                tqdm.write(f'FILE: {wav_file} NOT FOUND IN: {d}')
                                log_file.write(f'FILE: {wav_file} NOT FOUND IN: {d}\n')
                            except NonSpeciesException: # in case not able to extract species from annotation file
                                #not_extracted_counter += ann.shape[0]
                                l_not_extracted_counter[self.species2int('Unidentified')] += ann.shape[0]
                                species_not_identified_counter += ann.shape[0]
                                not_identified_species.append(filename)
                                tqdm.write(f'Not able to extract species and vocalization from: {filename}')
                                log_file.write(f'Not able to extract species and vocalization from: {filename}\n')
                                break

                        else: # event starts and ends in different files
                            #not_extracted_counter += 1
                            #multifile_event_counter += 1
                            species, vocalization = self.extract_species(filename)
                            l_not_extracted_counter[self.species2int(species)] += 1
                            l_multifile_event_counter[self.species2int(species)] += 1
                            tqdm.write(f'Event not extracted: Starts and ends in different files')
                            log_file.write(f'Event not extracted: Starts and ends in different files\n')
                            
        self.extraction_df.to_pickle(os.path.join(log_dir_path, 'extraction_df_'+ date_time_filename +'.pkl'))
        delta_time = datetime.now() - init_time
        self.print_extraction_report(log_file, l_extracted_counter, l_not_extracted_counter, species_not_identified_counter, l_file_not_found_counter, l_multifile_event_counter, l_non_positvide_duration_counter, not_identified_species, delta_time)
        log_file.close()

def main(dataset_path, output_path, min_frame_size_sec):
    extractor = Extractor(dataset_path, output_path, min_frame_size_sec)
    subdatasets = extractor.scan_dataset()
    extractor.extract(subdatasets)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Extract events from the specified dataset')

    parser.add_argument('dataset_path', type=str, help='path of the dataset to extract')
    parser.add_argument('output_path', type=str, help='directory where extracted events and log_file are saved')
    parser.add_argument('--min_frame_size_sec', type=int, default=0, help='minimum frame size of the extracted frames containing the events')

    params=parser.parse_args()

    main(params.dataset_path, params.output_path, params.min_frame_size_sec)