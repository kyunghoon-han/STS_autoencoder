from utils import Decoder, Encoder_Conv, ToText
from STS import batch_split
import librosa, torch, os
import numpy as np
import pickle
import soundfile as sf

def numtohangul(onehots, chars):
    dict_out = {}
    zipped = zip(onehots,chars)
    for a,b in zipped:
        counter = 0 
        for i in a:
            counter += 1
            if a == 1:
                break
            else:
                continue
        dict_out[counter] = b
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
epoch_num = 10
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
voice_num = 9987
input_wav = './preprocessing_unit/preprocessed/prep_data_wav9987.npz'
target_txt = './preprocessing_unit/preprocessed/prep_data_txt9987.npz'
hangul_dict_path = './preprocessing_unit/encoded_hangul.pkl'
with open(hangul_dict_path, 'rb') as f:
    hangul_dict = pickle.load(f)

# now load the npz files
txt = np.load(target_txt)['arr_0'].tolist()
wav = np.load(input_wav)['arr_0'].tolist()
txt = batch_split(txt,is_txt=True,dict_val=hangul_dict)
if len(txt) == 68:
    txt = txt[:-1]
elif len(txt) == 67:
    pass
else:
    print('this file is corrupted')
    exit()
txt = torch.FloatTensor(txt).to(device)
wav = torch.FloatTensor(batch_split(wav)).to(device)

# import a model
size_wav = wav.size()[-1]
size_txt = txt.size()[-1]
encoder = Encoder_Conv(device=device)
decoder = Decoder(size_wav,device=device,batch_size=batch_size)
to_text = ToText(output_size=size_txt,device=device)
checkpoint=torch.load(path_val)
encoder.load_state_dict(checkpoint['encoder_state'])
decoder.load_state_dict(checkpoint['decoder_state'])
to_text.load_state_dict(checkpoint['text_decoder_state'])
encoder.eval()
decoder.eval()
to_text.eval()

# now... roll it!
speech_result = decoder(encoder(wav))
speech_result = speech_result.reshape(-1,size_wav)
text_result = to_text(encoder(wav))
text_result = text_result.reshape(-1,size_txt).tolist()

# dictionary to bring the position of 1 in the one-hot to a hangul
reverse_dict = numtohangul(hangul_dict.values(), hangul_dict.keys())

# get the actual text...
text_stuff = ""
for a in text_result:
    count = 0
    for i in a:
        count += 1
        if i == 1:
            break
    text_stuff = text_stuff + reverse_dict[count]
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

