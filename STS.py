from utils import Decoder,Latent, LossMSE,Opt,Schedule,Encoder_Conv, ToText
import numpy as np
import librosa, torch, os, csv, pickle, shutil
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

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
batch_size = 4
epochs = 1000
num_test = 100

# ==================================

def batch_split(lst,batch_size=batch_size, is_txt=False, dict_val=None):
    # this splits the list_data above in batches
    # note that this function takes a list as an input
    list_tmp = [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]
    list_output = []
    for stuff in list_tmp:
        if isinstance(stuff, np.ndarray):
            stuff = np.ndarray.tolist(stuff)

        if len(stuff) == 4:
            list_output.append(stuff)
        elif len(stuff) < 4:
            if is_txt:
                while len(stuff) < 4:
                    stuff.append(dict_val['*'])
            else:
                while len(stuff) < 4:
                    stuff.append(0.0)
            list_output.append(stuff)
        else:
            print('some file has weird batch divisions...')
            continue
    return list_output

#====================================
def data_details(train_stuff,data_dir=data_dir):
    counter = 0
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
    
    # pre-define the models
    path_txt = ''
    path_wav = ''
    for pair in train_stuff:
        path_txt = os.path.join(data_dir.replace('preprocessed',''),pair[0].replace('./',''))
        path_wav = os.path.join(data_dir.replace('preprocessed',''),pair[1].replace('./',''))
        if not os.path.exists(path_txt):
            train_stuff.remove(pair)
            continue
        if not os.path.exists(path_wav):
            train_stuff.remove(pair)
            continue
        break
    txt_target = np.load(path_txt)['arr_0'].tolist()
    wav_source = np.load(path_wav)['arr_0'].tolist()
    txt_target = torch.FloatTensor(batch_split(txt_target,
                                    is_txt=True,
                                    dict_val=dict_txt)).to(device)
    wav_source = torch.FloatTensor(batch_split(wav_source)).to(device) 
    output_size = wav_source.size()[-1]
    encoder = Encoder_Conv(device=device) # encoder module
    input_size = encoder(wav_source).size()[-1]
    latent_1 = Latent(input_size,device) # latent layers for the wav decoder
    decoder = Decoder(input_size,output_size,device=device,batch_size=batch_size) # decoder module
    opt_1 = Opt(encoder,adam=False,learning_rate=0.5) # optimizers
    opt_2 = Opt(decoder,learning_rate=0.1)
    opt_l1 = Opt(latent_1, learning_rate=0.5)
    sch_1 = Schedule(opt_1,step_size=100)
    sch_2 = Schedule(opt_2,step_size=100)
    sch_l1 = Schedule(opt_l1, step_size=100)
    output_size = txt_target.size()[-1]
    latent_2 = Latent(input_size,device)
    text_decoder = ToText(input_size,output_size=output_size,device=device) # to_text module
    opt_l2 = Opt(latent_2, learning_rate=0.5)
    opt_t = Opt(text_decoder,adam=False, learning_rate=0.05)
    sch_t = Schedule(opt_t, step_size=100)
    sch_l2 = Schedule(opt_l2,step_size=100)


    while e < epochs :
        e += 1
        num_discarded = 0
        print("Epoch: ",e)
        for pair in tqdm(train_stuff):
            # load the data
            path_txt = os.path.join(data_dir.replace('preprocessed',''),pair[0].replace('./',''))
            path_wav = os.path.join(data_dir.replace('preprocessed',''),pair[1].replace('./',''))
            if not os.path.exists(path_txt):
                train_stuff.remove(pair)
                num_discarded += 1
                continue
            elif not os.path.exists(path_wav):
                train_stuff.remove(pair)
                num_discarded += 1
                continue
            txt_target = np.load(path_txt)['arr_0'].tolist()
            wav_source = np.load(path_wav)['arr_0'].tolist()
            
            # split the data into batches
            tmp = batch_split(txt_target,
                    is_txt=True,
                    dict_val=dict_txt)
            if len(tmp) == 68:
                tmp = tmp[:-1]
            elif len(tmp) == 67:
                pass
            else:
                print("a text file is corrupted")
                continue
            txt_target = torch.FloatTensor(tmp).to(device)
            wav_source = torch.FloatTensor(batch_split(wav_source)).to(device)

            # an STS autoencoder module
            opt_1.zero_grad()
            opt_2.zero_grad()
            opt_l1.zero_grad()
            encoded_wav = encoder(wav_source)
            latent_output = latent_1(encoded_wav)
            decoded_wav = decoder(latent_output.to(device))
            loss_STS = LossMSE(decoded_wav,wav_source.to(device),device,switch=True)
            loss_STS.backward()
            opt_2.step()
            opt_l1.step()
            opt_1.step()
            
            # an STT module
            opt_t.zero_grad()
            opt_l2.zero_grad()
            encoded_wav = encoder(wav_source).detach()
            latent_output = latent_2(encoded_wav)
            text_source = text_decoder(latent_output)
            loss_STT = LossMSE(text_source,txt_target,device)
            loss_STT.backward()
            opt_t.step()
            opt_l2.step()

            if sch_1 is not None:
                sch_1.step()
                sch_2.step()
                sch_t.step()
                sch_l1.step()
                sch_l2.step()
            else:
                print("The schedulers are not properly defined")

            loss = loss_STS + loss_STT

            if step % 3000 == 7:
                string_out = "At step %i, the loss is %.4f" % (step, loss)
                print(string_out)
                string_out = "with the STT loss %.4f, STS loss %.4f" % (loss_STT, loss_STS)
                print(string_out)
            
            if step % 100 == 0:
                string_out = "%i , %.3f, %.3f, %.3f \n" % (step, loss, loss_STT,loss_STS)
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
            #wav_source = std_wav.transform(wav_source)
            # split the data into batches
            tmp = batch_split(txt_target,
                    is_txt=True,
                    dict_val=dict_txt)
            if len(tmp) == 68:
                tmp = tmp[:-1]
            elif len(tmp) == 67:
                pass
            else:
                print("a text file is corrupted")
                continue
            txt_target = torch.FloatTensor(tmp).to(device)
            wav_source = torch.FloatTensor(batch_split(wav_source)).to(device)
            #if wav_source.size()[0] != txt_target.size()[0]:
            #    continue
            # just to make sure the the input values are floats
            txt_target = txt_target.float()
            wav_source = wav_source.float()
            #STT
            encoded_wav = encoder(wav_source)
            txt_source = text_decoder(encoded_wav)
            loss_2 = LossMSE(txt_source,txt_target,device)
            # STS
            encoded_wav = encoder(wav_source)
            decoded_wav = decoder(encoded_wav)
            loss_1 = LossMSE(decoded_wav,wav_source,device,switch=True)

            test_loss = test_loss + loss_1+loss_2
            counter += 1.0
        test_loss = test_loss
        test_loss = test_loss / counter
        print("The number of discarded data is: ",num_discarded)
        test_report = "The validation loss is : %.5f" %(test_loss)
        print(test_report)
        # to save the model...
        if e%1 == 0:
            if not os.path.exists("./models"):
                os.mkdir("./models")
            path = "./models/epoch_%i.pt" %(e)
            torch.save({
                'epoch' : e,
                'encoder_state' : encoder.state_dict(),
                'decoder_state' : decoder.state_dict(),
                'text_decoder_state': text_decoder.state_dict(),
                'latent_wav_state' : latent_1.state_dict(),
                'latent_txt_state' : latent_2.state_dict(),
                #'optimizer_state_encoder': opt_1.state_dict(),
                #'optimizer_state_attention': opt_2.state_dict(),
                #'optimizer_state_decoder': opt_d.state_dict(),
                'loss' : loss}, path)
    f.close()

if __name__=='__main__':
    trainer()
