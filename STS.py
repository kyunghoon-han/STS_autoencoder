from utils import Encoder_GRU, AttentionDecoder, Decoder,LossKL, LossMSE,LossBCEMSE, Opt, Schedule 
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
    f.write("step, loss, STT_loss, STS_loss \n")
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

            # an STT module
            if step == 0 :
                input_size = wav_source.size()[-1]
                encoder = Encoder_GRU(input_size=input_size,
                                    hidden_size=input_size,
                                    device=device) # call the models
                opt_1 = Opt(encoder) # optimizers
                #sch_1 = Schedule(opt_1)
            # optimizer initialization and encoder definition
            opt_1.zero_grad()
            encoded_wav, hidden = encoder(wav_source)

            opt_1.zero_grad()
            encoded_wav,hidden = encoder(wav_source)

            if step == 0:
                input_size = encoded_wav.size()[-1]
                hidden_size= hidden.size()[-1]
                attention = AttentionDecoder(hidden_size = input_size,
                                             output_size = 199,
                                             vocab_size=txt_target.size()[-1],
                                             device=device)
                opt_2 = Opt(attention)
                #sch_2 = Schedule(opt_2)
            opt_2.zero_grad()
            att_output, att_hidden, normalized_weights = attention(
                                    hidden.to(device),
                                    encoded_wav.to(device))

            loss_1 = LossBCEMSE(att_output,txt_target)
            # backprogation
            loss_1.backward()
            opt_2.step()
            opt_1.step()

            # Now to the STS model

            if step == 0:
                output_size = wav_source.size()[-1]
                decoder = Decoder(output_size,device=device,batch_size=batch_size)
                opt_d = Opt(decoder)
                #sch_d = Schedule(opt_d)
            opt_d.zero_grad()
            opt_1.zero_grad()
            opt_2.zero_grad()
            
            encoded_wav,hidden = encoder(wav_source)
            att_output, att_hidden, normalized_weights = attention(
                                    hidden.to(device),
                                    encoded_wav.to(device))
            decoded_wav = decoder(att_output)

            loss_2 = LossKL(decoded_wav,wav_source.to(device)) + LossMSE(decoded_wav,wav_source.to(device))
            loss_2.backward()
            opt_1.step()
            opt_2.step()
            opt_d.step()

            loss = loss_1 + loss_2
            if step % 1000 == 1:
                string_out = "At step %i, the loss is %.2f" % (step, loss)
                print(string_out)
                string_out = "with the STT loss %.2f, STS loss %.2f" % (loss_1, loss_2)
                #sch_1.step()
                #sch_d.step()
                #sch_2.step()
        
            
            if step % 100 == 0:
                string_out = "%i , %.2f, %.2f, %.2f \n" % (step, loss, loss_1,loss_2)
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
            #STT
            encoded_wav, hidden = encoder(wav_source)
            result_att, _,_ = attention(hidden,encoded_wav)
            loss_2 = LossBCEMSE(result_att,txt_target)
            # STS
            encoded_wav, hidden = encoder(wav_source)
            result_att, _,_ = attention(hidden,encoded_wav)
            decoded_wav = decoder(result_att)
            loss_1 = LossKL(decoded_wav,wav_source.to(device)) 

            test_loss = test_loss + loss_1+loss_2
            counter += 1
        test_loss = test_loss / counter
        print("The number of discarded data is: ",num_discarded)
        print("The validation loss is : ", test_loss)
        #sch_1.step()
        #sch_d.step()
        #sch_2.step()
        # to save the model...
        if e%10 == 0:
            if not os.path.exists("./models"):
                os.mkdir("./models")
            path = "./models/epoch_%i.pt" %(e)
            torch.save({
                'epoch' : e,
                'encoder_state' : encoder.state_dict(),
                'decoder_state' : decoder.state_dict(),
                'attention_state': attention.state_dict(),
                #'optimizer_state_encoder': opt_1.state_dict(),
                #'optimizer_state_attention': opt_2.state_dict(),
                #'optimizer_state_decoder': opt_d.state_dict(),
                'loss' : loss}, path)
    f.close()

if __name__=='__main__':
    trainer()
