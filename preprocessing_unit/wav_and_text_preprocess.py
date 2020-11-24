import os, csv
import librosa
import numpy as np
from text_preprocess import sentence2vec
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

def data_dict(file_infos):
    return [match_sentence(a) for a in file_infos]

# =================================================
#
# Read and process the WAV files
#
def wav_list_from_data_dict_result(data_dict_result,directory = './preprocessed',output_file_name='prep_data'):
    file_root_name = os.path.join(directory,output_file_name)
    list_tmp = []
    len_max = 0
    sr_max = 0
    '''
    print("Get the longest text length")
    for speaker in tqdm(data_dict_result):
        for speaker_utterances in speaker:
            wave_file = speaker_utterances[0]
            if os.path.isfile(wave_file):
                wav, sr = librosa.load(wave_file,sr=None)
                if len(wav) > len_max:
                    len_max = len(wav)
                if sr > sr_max:
                    sr_max = sr
    '''
    counter = 0
    sp_num = 0
    print("Obtain the wav vector & text pairs")
    for speaker in tqdm(data_dict_result):
        print("Speaker number: ", sp_num)
        sp_num += 1
        for speaker_utterances in tqdm(speaker):
            # we will save 1 npz file per voice data
            wav_filename = file_root_name+"_wav"+str(counter)+".npz"
            txt_filename = file_root_name+"_txt"+str(counter)+".npz"
            wav_file = speaker_utterances[0]
            sentence = speaker_utterances[1]
            sentence = sentence.replace("“","\"").replace("”","\"").replace("/"," ")
            sentence = sentence.replace("’","'").replace("‘","'").replace("`","'")
            sentence = sentence.replace("@","*").replace("#","*").replace(":","*")
            sentence = sentence.replace("+","*").replace("-","*").replace("=","*")
            sentence = sentence.replace("~","*").replace("&","*").replace("%","*")
            sentence = sentence.replace("[","(").replace("{","(").replace("#","*")
            sentence = sentence.replace("]",")").replace("}",")").replace("_","*")
            if os.path.isfile(wav_file):
                wav, sr = librosa.load(wav_file,sr=None)
                # obtain the MEL and save the array to the respective file
                mel = librosa.feature.melspectrogram(y=wav, sr=sr)
                np.savez(wav_filename, mel)
                # encode the sentence
                if counter == 0:
                    encoded_sentence, hangul_dict = sentence2vec(sentence)
                else:
                    encoded_sentence, hangul_dict = sentence2vec(sentence, hangul_dict=hangul_dict)

                np.savez(txt_filename, encoded_sentence)
                counter = counter + 1 # update the counter

# ========================================
#
#  Main part of the code
#
if __name__=='__main__':
    # So this variable contains all the information I need
    # on the data to process
    file_infos = the_wavs(data_directory)
    paths_and_sentences = data_dict(file_infos)
    wav_list_from_data_dict_result(paths_and_sentences)
