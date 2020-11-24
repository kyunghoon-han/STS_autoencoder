import os, csv, random
import librosa
import numpy as np
from tqdm import tqdm

#
#   Some definitions
#
data_directory = "./data"

def the_wavs(dir_in):
    list_dirs = []
    tmp_list = []
    counter = 0
    for (dirpath, dirnames, _) in os.walk(dir_in):
        for direc in dirnames:
            direc = os.path.join(dirpath,direc)
            list_files = list_paths(direc)
            list_text = list_paths(direc,extension='.csv')
            tmp_list = [direc,list_text[0],list_files]
            list_dirs.append(tmp_list)

    return list_dirs


def list_paths(dirpath, extension='.wav'):
    # get the paths of the files (preferably a wav file) in the
    # input directory path
    # - This returns a list of file paths
    output = []
    for filename in os.listdir(dirpath):
        if filename.endswith(extension):
            output.append(filename)
    return output

# So this variable contains all the information I need
# on the data to process
file_infos = the_wavs(data_directory)

#
# Match each sentences to the relevant file
#
def csv_to_dict(csv_path):
    dict_out = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            dict_out[row[0]] = row[1]
    return dict_out

def match_sentence(file_info, extension='.wav'):
    # This takes a file info from file_infos defined above
    # and returns a list of .wav file path and text pairs
    dir_path = file_info[0]
    text_dict = csv_to_dict(os.path.join(dir_path,file_info[1]))
    path_wavs = [ os.path.join(dir_path,a) for a in file_info[2]]
    list_wav_names = [a.replace('.wav','') for a in file_info[2]]
    
    wavs = zip(path_wavs, list_wav_names)
    list_out = []
    for (path_ut, utterance) in wavs:
        try:
            text = text_dict[utterance]
            list_out.append([path_ut,text])
        except KeyError:
            continue
    return list_out

def data_dict(input_file_infos):
    return [match_sentence(a) for a in input_file_infos]

def flatten_data_dict(data_dict_in, shuffle = True):
    flat_list = []
    for sublist in data_dict_in:
        for subsublist in sublist:
            flat_list.append(subsublist)
    if shuffle:
        random.shuffle(flat_list)
    return flat_list

# =================================================

# This now contains the list of data
# paths_and_sentences[i] = list of data of an i-th speaker
# paths_and_sentences[i][j] = a filename, sentence pair of an i-th speaker
# paths_and_sentences[i][j][0] = a filename
# paths_and_sentences[i][j][1] = a sentence
#paths_and_sentences = data_dict(the_wavs)

# speaker-independent list of filename, sentence pairs
#flat_stuff = flatten_data_dict(paths_and_sentences)

# ================================================
# now we will use this function to call the data
# for the main module
def data_caller(data_dir):
    return flatten_data_dict(data_dict(the_wavs(data_dir)))
