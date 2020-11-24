from utils import data_caller
import numpy as np
import librosa
import torch

# data directory
data_dir = "./data"
# list of filename, sentence pairs
list_data = data_caller(data_dir)

#====================================
# Hyperparameters 
batch_size = 4

def batch_split(lst,batch_size=batch_size):
    # this splits the list_data above in batches
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]

#=====================================
# Audio utils

def wav2mel(audio_np):
    S = librosa.feature.melspectrogram(y=audio_np)
    return S

def max_len_n_sr(list_pairs):
    max_len = 0.0
    counter = 0
    mean_sr = 0
    for stuff in list_pairs:
        counter = counter + 1
        audio, sr = librosa.load(stuff[0])
        mean_sr += sr
        if len(audio) > max_len:
            max_len = len(audio)
    mean_sr = mean_sr / counter
    return max_len, mean_sr

def mels4each_batch(list_pairs):
    list_tmp = []
    for stuff in list_pairs:
        audio, sr = librosa.load(stuff[0])
        audio = audio.tolist()
        max_len, _ = max_len_n_sr(list_pairs)
        while len(audio) < max_len:
            audio.append(0.0)
        list_tmp.append(wav2mel(np.array(audio)).tolist())
    output_temp = torch.transpose(torch.FloatTensor(list_tmp),1,2)
    return torch.view(output_temp, 

# list of filenames and sentences
list_in = batch_split(list_data)
ahhhh = mels4each_batch(list_in[0])
print(ahhhh.size())

#=====================================

