from utils import Encoder_GRU, AttentionDecoder, Loss, Opt 
import numpy as np
import matplotlib.pyplot as plt
import librosa, torch, os, csv, pickle, shutil
from tqdm import tqdm

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# preprocessed data directory
data_dir = "./preprocessing_unit/preprocessed"
encoded_txt_pkl = 'encoded_hangul.pkl'
dict_txt_encoder = os.path.join(data_dir.replace("preprocessed",""), encoded_txt_pkl)
path_list_pairs = os.path.join(data_dir.replace("preprocessed",""), 'filelist_pairs.csv')

def open_list_pairs(path=path_list_pairs):
    # this outputs the list of filenames
    list_out = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            stuff = [row[0],row[1]]
            list_out.append(stuff)
    return list_out

#====================================
# Hyperparameters 
batch_size = 64
epochs = 100
num_test = 100

# ==================================

def batch_split(lst,batch_size=batch_size, is_txt=False, dict_val=None):
    # this splits the list_data above in batches
    # note that this function takes a list as an input
    list_tmp = [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]
    list_output = []
    for stuff in list_tmp:
        if len(stuff) == 64:
            list_output.append(stuff)
        elif len(stuff) < 64:
            if is_txt:
                while len(stuff) < 64:
                    stuff.append(dict_val['*'])
            else:
                while len(stuff) < 64:
                    stuff.append(0.0)
            list_output.append(stuff)
        else:
            print('some file has weird batch divisions...')
            continue
    return list_output



#=====================================
#
#    Trainer
#
#=====================================
def trainer():
    # first we have to call the data in
    list_stuff = open_list_pairs()
    test_stuff = list_stuff[-100:]
    train_stuff = list_stuff[:-100]
    with open(dict_txt_encoder, 'rb') as f:
        dict_txt = pickle.load(f) # loads the hangul encoder
    # call the models
    e = 0 # epoch
    # log directory and file
    log_dir = "./logs"
    log_filename = "logs.csv"
    log_path = os.path.join(log_dir, log_filename)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)
    else:
        os.mkdir(log_dir)
    # model directory
    if os.path.exists("./models"):
        shutil.rmtree("./models")
        os.mkdir("./models")

    f = open(log_path,"w+")
    step = 0
    num_discarded = 0
    f.write("step, loss \n")
    while e < epochs :
        e += 1
        num_discarded = 0
        print("Epoch: ",e)
        for pair in tqdm(train_stuff):
            # load the data
            path_txt = os.path.join(data_dir.replace('preprocessed',''),pair[0].replace('./',''))
            path_wav = os.path.join(data_dir.replace('preprocessed',''),pair[1].replace('./',''))
            txt_target = np.load(path_txt)['arr_0'].tolist()
            wav_source = np.load(path_wav)['arr_0'].tolist()
            # split the data into batches
            txt_target = torch.FloatTensor(batch_split(txt_target,
                                            is_txt=True,
                                            dict_val=dict_txt)).to(device)
            wav_source = torch.FloatTensor(batch_split(wav_source)).to(device)
            # discard the long sources
            if wav_source.size()[0] != txt_target.size()[0]:
                step -= 1
                num_discarded += 1
                continue
            
            if step == 0 :
                input_size = wav_source.size()[-1]
                encoder = Encoder_GRU(input_size=input_size,
                                    hidden_size=input_size,
                                    device=device) # call the models
                opt_1 = Opt(encoder) # optimizers
            # optimizer initialization and encoder definition
            opt_1.zero_grad()
            encoded_wav, hidden = encoder(wav_source)
            
            if step == 0:
                input_size = encoded_wav.size()[-1]
                hidden_size= hidden.size()[-1]
                attention = AttentionDecoder(hidden_size = input_size,
                                             output_size = 199,
                                             vocab_size=txt_target.size()[-1],
                                             device=device)
                opt_2 = Opt(attention)
            att_output, att_hidden, normalized_weights = attention(
                                    hidden.to(device),
                                    encoded_wav.to(device))

            loss = Loss(att_output,txt_target)
            # backprogation
            loss.backward()
            opt_2.step()
            opt_1.step()

            if step % 1000 == 0:
                string_out = "At step %i, the loss is %.2f" % (step, loss)
                print(string_out)
            
            if step % 100 == 0:
                string_out = "%i , %.2f\n" % (step, loss)
                f.write(string_out)
            step += 1
        # now to test the model...
        test_loss = 0.0
        counter = 0.0
        for pair in test_stuff:
            path_txt = os.path.join(data_dir.replace('preprocessed',''),pair[0].replace('./',''))
            path_wav = os.path.join(data_dir.replace('preprocessed',''),pair[1].replace('./',''))
            txt_target = np.load(path_txt)['arr_0'].tolist()
            wav_source = np.load(path_wav)['arr_0'].tolist()
            # split the data into batches
            txt_target = torch.FloatTensor(batch_split(txt_target,
                                            is_txt=True,
                                            dict_val=dict_txt)).to(device)
            wav_source = torch.FloatTensor(batch_split(wav_source)).to(device)
            if wav_source.size()[0] != txt_target.size()[0]:
                continue
            # just to make sure the the input values are floats
            txt_target = txt_target.float()
            wav_source = wav_source.float()
            encoded_wav, hidden = encoder(wav_source)
            result_att, _,_ = attention(hidden,encoded_wav)
            loss = Loss(result_att,txt_target)
            test_loss += loss
        test_loss = test_loss / counter
        print("The number of discarded data is: ",num_discarded)
        print("The validation loss is : ", test_loss)
        # to save the model...
        if e%10 == 0:
            if not os.path.exist("./models"):
                os.mkdir("./models")
            path = "./models/epoch_%i.pt" %(e)
            torch.save({
                'epoch' : e,
                'encoder_state' : encoder.state_dict(),
                'optimizer_state': opt.state_dict(),
                'loss' : loss}, path)
    f.close()

if __name__=='__main__':
    trainer()
