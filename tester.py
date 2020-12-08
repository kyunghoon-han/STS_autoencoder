from utils import Latent, Decoder, Encoder_Conv, ToText
from STS import batch_split
import librosa, torch, os
import numpy as np
import pickle
import soundfile as sf

def numtohangul(onehots, chars,is_num=True):
    # is_num=True : hangul-to-numeric value encoding used
    # otherwise, one-hot-encoding is used
    dict_out = {}
    zipped = zip(onehots,chars)
    for a,b in zipped:
        if not is_num:
            counter = 0
            for i in a:
                counter += 1
                if a == 1:
                    break
                else:
                    continue
            dict_out[counter] = b
        else:
            dict_out[a] = b
    return dict_out

def nan_to_0(array2d_in):
    list_out= []
    for tmp in array2d_in:
        list_tmp = []
        for a in tmp:
            if a==np.nan:
                list_tmp.append(0.0)
            else:
                list_tmp.append(a)
        list_out.append(list_tmp)
    return list_out

batch_size = 4
# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model directory
epoch_num = 3
filename = 'epoch_'+str(epoch_num)+'.pt'
model_dir = './models'
path_val = os.path.join(model_dir,filename)
#save tests to...
test_dir = './tests'
if os.path.exists(test_dir):
    pass # don't have to remove the directory for now...
else:
    os.mkdir(test_dir) # make the test dir if it's not already there

# input data path
voice_num = 44
input_wav = './preprocessing_unit/preprocessed/prep_data_wav44.npz'
target_txt = './preprocessing_unit/preprocessed/prep_data_txt44.npz'
hangul_dict_path = './preprocessing_unit/encoded_hangul.pkl'
with open(hangul_dict_path, 'rb') as f:
    hangul_dict = pickle.load(f)

# now load the npz files
txt = np.load(target_txt)['arr_0'].tolist()
wav = np.load(input_wav)['arr_0'].tolist()
txt = batch_split(txt,is_txt=True,dict_val=hangul_dict)
if len(txt) >34:
    txt = txt[:34]
elif len(txt) == 34:
    pass
else:
    print('this file is corrupted')
    exit()
txt = torch.FloatTensor(txt).to(device)
wav = torch.FloatTensor(batch_split(wav)).to(device)

# import a model
checkpoint=torch.load(path_val)
size_wav = wav.size()[-1]
size_txt = txt.size()[-1]
encoder = Encoder_Conv(device=device)
encoder_out = encoder(wav)
latent_in = encoder_out.size()[-1]
latent_1 = Latent(latent_in,device=device)
latent_2 = Latent(latent_in,device=device)
encoder.load_state_dict(checkpoint['encoder_state'])
latent_1.load_state_dict(checkpoint['latent_wav_state'])
latent_2.load_state_dict(checkpoint['latent_txt_state'])
latent_out = latent_1(encoder(wav))
size_in = latent_out.size()[-1]
decoder = Decoder(size_in,output_size=size_wav,device=device,batch_size=batch_size)
to_text = ToText(size_in,output_size=1,device=device)
decoder.load_state_dict(checkpoint['decoder_state'])
to_text.load_state_dict(checkpoint['text_decoder_state'])
encoder.eval()
decoder.eval()
to_text.eval()

# now... roll it!
speech_result = decoder(latent_1(encoder(wav)))
speech_result = speech_result.reshape(-1,size_wav)
text_result = to_text(latent_2(encoder(wav)))
text_result = text_result.reshape(-1,size_txt).tolist()

# dictionary to bring the position of 1 in the one-hot to a hangul
reverse_dict = numtohangul(hangul_dict.values(), hangul_dict.keys())

# get the actual text...
text_stuff = ""
'''
for a in text_result:
    count = 0
    for i in a:
        count += 1
        if i == 1:
            break
    text_stuff = text_stuff + reverse_dict[count]
'''
for a in text_result:
    print(a)
    exit()
    b = round(a)
    text_stuff = text_stuff + reverse_dict[b]
text_stuff.replace('*','')
print(text_stuff)
    

# get the sound data
def round_list(x):
    return [round(a) for a in x]
speech_result = speech_result.detach().cpu().numpy()
speech_result = np.array(nan_to_0(speech_result))
#speech_result = [round_list(x) for x in speech_result]

#speech_result = librosa.core.db_to_power(speech_result)
speech_result = librosa.feature.inverse.mel_to_audio(speech_result)
# and output the data to a WAV file
sf.write(os.path.join(test_dir,str(voice_num)+'.wav'),
                        speech_result,
                        22050)

