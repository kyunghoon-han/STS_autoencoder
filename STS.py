import numpy as np
from utils import Decoder,Latent,Criterion,Opt,Schedule,Encoder_Conv, TEncoder,TDecoder, TLatent
import librosa, torch, os, csv, pickle, shutil, random
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
#batch_size = 64
#epochs = 1000
#num_test = 100
#lr_stt = 0.01
#lr_tts = 0.0001
#step_lrtts = 20
#step_lrstt = 10
#STS_threshold = 10
#verbose = 5
# ===================================

def batch_split(lst,batch_size, is_txt=False, dict_val=None):
    # this splits the list_data above in batches
    # note that this function takes a list as an input
    list_tmp = [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]
    list_output = []
    for stuff in list_tmp:
        if isinstance(stuff, np.ndarray):
            stuff = np.ndarray.tolist(stuff)

        if len(stuff) == batch_size:
            list_output.append(stuff)
        elif len(stuff) < batch_size:
            if is_txt:
                while len(stuff) < batch_size:
                    stuff.append(dict_val['*'])
            else:
                while len(stuff) < batch_size:
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
        txt_target = torch.FloatTensor(batch_split(txt_target,batch_size,
                                is_txt=True,
                                dict_val=dict_txt)).to(device)
        wav_source = torch.FloatTensor(batch_split(wav_source,batch_size)).to(device)
        # discard the long sources
        if wav_source.size()[0] != txt_target.size()[0]:
            step -= 1
            num_discarded += 1
            continue

def text_pruner(text,leng_threshold):
    if len(text) > leng_threshold:
        text = text[:leng_threshold]
        return text
    elif len(text) == leng_threshold:
        return text
    else:
        print('a text file is corrupted', len(tmp))
        raise

#=====================================
#
#    Trainer
#
#=====================================
def trainer(batch_size=64, epochs=1000,num_test=100,
            lr_stt = 0.01, lr_tts=0.0001, step_lrtts=20,
            step_lrstt=10, STS_threshold = 10, verbose = 5,
            data_dir = "./preprocessing_unit/preprocessed",
            encoded_txt_pkl = 'encoded_hangul.pkl',
            log_dir="./logs", log_filename="logs.csv",
            model_save_dir="./models",device='cuda',device_auto=True):
    '''
        input information:
            batch_size      : number of batches
            epochs
            num_test        : number of test data
            lr_stt          : learning rate of STT parts of the model
            lr_tts          : learning rate of TTS parts of the model
            step_lrtts      : scheduler step size for the TTS parts
            step_lrstt      : scheduler step size for the STT parts
            STS_threshold   : the epoch to start combining TTS and STT parts to
                              STS and TTT models
            vebose          : verbosity of the model
            data_dir        : directory where the preprocessed data are stored 
                              (file formats: .npz or .npy)
            encoded_txt_pkl : a pickle file with text-to-one-hot dictionary
            log_dir         : directory to store the log file
            log_filename    : name of the log file
            model_save_dir  : directory to store the models
            device          : the device name where the NNs will be run on
            device_auto     : this is true if the user want the program to find 
                              the device to run the program on
    '''
    # set device if device_auto=True
    if device_auto:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = device # just a filler to not confuse myself
    dict_txt_encoder = os.path.join(data_dir.replace("preprocessed",""), encoded_txt_pkl)
    path_list_pairs = os.path.join(data_dir.replace("preprocessed",""), 'filelist_pairs.csv')
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
    f.write("step, loss, STT_loss, TTS_loss \n")
    
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
    # target text input
    txt_target = np.load(path_txt)['arr_0'].tolist()
    # length of the batch_devided txt input (modified again later)
    length_txt = len(txt_target) * len(txt_target[0])
    # source audio
    wav_source = np.load(path_wav)['arr_0'].tolist()
    txt_target = torch.FloatTensor(batch_split(txt_target,
                                    batch_size,is_txt=True,
                                    dict_val=dict_txt)).to(device)
    # modification of the length_txt
    length_txt = int(length_txt/(txt_target.size()[-1] * batch_size))
    txt_target = text_pruner(txt_target,length_txt) # prune the input
    wav_source = torch.FloatTensor(batch_split(wav_source,batch_size)).to(device) 
    output_size = wav_source.size()[-1]
    encoder = Encoder_Conv(device=device,input_dim=wav_source.size()[-1]) # encoder module
    text_encoder = TEncoder(device=device) # text encoder module
    input_size = encoder(wav_source).size()[-1]
    output_size = txt_target.size()[-1]
    latent_1 = Latent(input_size,device) # latent layers for the wav decoder
    l_out = latent_1(encoder(wav_source))
    latent_size = l_out.size()[-1]
    decoder = Decoder(latent_size,output_size,device=device,batch_size=batch_size) # decoder module
    # Optimizers 
    opt_1 = Opt(encoder,learning_rate=lr_tts)
    opt_te = Opt(text_encoder,learning_rate=lr_stt) 
    opt_2 = Opt(decoder,learning_rate=lr_tts)
    opt_l1 = Opt(latent_1, learning_rate=lr_tts)
    # Schedulers
    sch_1 = Schedule(opt_1,step_size=lr_tts)
    sch_te = Schedule(opt_te,step_size=lr_stt)
    sch_2 = Schedule(opt_2,step_size=step_lrtts)
    sch_l1 = Schedule(opt_l1, step_size=step_lrtts)
    # optimizers and schedulers for the text_decoder and the second latent network
    output_size = wav_source.size()[-1]
    input_size = text_encoder(txt_target.to(device)).size()[-1]
    latent_2 = TLatent(input_size,device)
    text_decoder = TDecoder(latent_size,output_size,device=device,batch_size=batch_size)
    #ToText(latent_size,output_size=txt_target.size()[-1],device=device) # to_text module
    opt_l2 = Opt(latent_2, learning_rate=lr_stt)
    opt_t = Opt(text_decoder,learning_rate=lr_stt)
    sch_t = Schedule(opt_t, step_size=step_lrstt)
    sch_l2 = Schedule(opt_l2,step_size=step_lrstt)


    while e < epochs :
        e += 1
        num_discarded = 0
        print("Epoch: ",e)
        random.shuffle(train_stuff)
        switch_counter = 0
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
            #txt_target = librosa.util.normalize(txt_target)
            wav_source = librosa.util.normalize(wav_source)

            # split the data into batches
            tmp = batch_split(txt_target,
                    batch_size,is_txt=True,
                    dict_val=dict_txt)
            tmp = text_pruner(tmp,length_txt)

            txt_target = torch.FloatTensor(tmp).to(device)
            wav_source = torch.FloatTensor(batch_split(wav_source,batch_size)).to(device)

            if e > STS_threshold:
                switch = True
            else:
                switch = False

            if not switch:
                #
                # First train an STT module
                #
                opt_1.zero_grad()
                opt_2.zero_grad()
                opt_l1.zero_grad()
                encoded_wav = encoder(wav_source)
                encoded_txt = text_encoder(txt_target).detach()
                latent_wav = latent_1(encoded_wav)
                latent_txt = latent_2(encoded_txt).detach()
                decoded_wav, loss_STT = decoder(latent_wav.to(device),
                        latent_txt, txt_target)
                loss_STT.backward()
                opt_2.step()
                opt_l1.step()
                opt_1.step()
                #
                # then train an TTS module
                #
                opt_te.zero_grad()
                opt_t.zero_grad()
                opt_l2.zero_grad()
                encoded_wav = encoder(wav_source).detach()
                encoded_txt = text_encoder(txt_target)
                latent_wav = latent_1(encoded_wav).detach()
                latent_txt = latent_2(encoded_txt)
                decoded_wav, loss_TTS = text_decoder(latent_txt.to(device),
                        latent_wav,wav_source)
                loss_TTS.backward()
                opt_t.step()
                opt_te.step()
                opt_l2.step()
            else:
                # get the fake stuff
                encoded_wav = encoder(wav_source).detach()
                encoded_txt = text_encoder(txt_target).detach()
                latent_wav = latent_1(encoded_wav).detach()
                latent_txt = latent_2(encoded_txt).detach()
                txt_fake, _ = decoder(latent_wav.to(device),
                        latent_txt, txt_target,backprop=False)
                wav_fake, _ = text_decoder(latent_txt.to(device),
                        latent_wav,wav_source,backprop=False)
                txt_fake = txt_fake.detach()
                wav_fake = wav_fake.detach()
                # now to TTT module
                opt_t.zero_grad()
                opt_te.zero_grad()
                opt_l2.zero_grad()
                encoded_wav = encoder(wav_source).detach()
                encoded_txt = text_encoder(txt_target)
                latent_wav = latent_1(encoded_wav).detach()
                latent_txt = latent_2(encoded_txt)
                decoded_text, loss_TTS = text_decoder(latent_txt.to(device),
                        latent_wav, wav_fake)
                loss_TTS.backward()
                tts_tmp = loss_TTS.item()
                opt_t.step()
                opt_te.step()
                opt_l2.step()
                opt_1.zero_grad()
                opt_2.zero_grad()
                opt_l1.zero_grad()
                decoded_text = decoded_text.detach()
                post_wav = latent_1(encoder(decoded_text))
                latent_txt = latent_2(text_encoder(txt_target))
                decoded_wav, loss_STT = decoder(post_wav,
                        latent_txt,
                        txt_target)
                loss_STT.backward()
                decoded_wav = decoded_wav.detach()
                stt_tmp = loss_STT.item()
                opt_1.step()
                opt_2.step()
                opt_l1.step()
                # now to STS module
                opt_1.zero_grad()
                opt_2.zero_grad()
                opt_l1.zero_grad()
                encoded_wav = encoder(wav_source)
                encoded_txt = text_encoder(txt_target).detach()
                latent_txt = latent_2(encoded_txt).detach()
                latent_wav = latent_1(encoded_wav)
                decoded_wav, loss_STT = decoder(latent_wav,latent_txt,txt_fake)
                loss_STT.backward()
                opt_1.step()
                opt_2.step()
                opt_l1.step()
                opt_1.zero_grad()
                opt_2.zero_grad()
                opt_l2.zero_grad()
                decoded_wav = decoded_wav.detach()
                post_txt = latent_2(text_encoder(decoded_wav))
                latent_wav = latent_wav.detach()
                decoded_txt, loss_TTS = text_decoder(post_txt,latent_wav,wav_source)
                loss_TTS.backward()
                opt_1.step()
                opt_2.step()
                opt_l2.step()

                loss_STT = 0.5*(loss_STT.item() + stt_tmp)
                loss_TTS = 0.5*(loss_TTS.item() + tts_tmp)

            if sch_1 is not None:
                sch_1.step()
                sch_2.step()
                sch_t.step()
                sch_l1.step()
                sch_l2.step()
            else:
                print("The schedulers are not properly defined")


            loss = loss_TTS + loss_STT

            if step % (verbose*100) == 0:
                string_out = "At step %i, the loss is %.6f" % (step, loss)
                print(string_out)
                string_out = "with the STT loss %.6f, TTS loss %.6f" % (loss_STT, loss_TTS)
                print(string_out)
            
            if step % verbose == 0:
                string_out = "%i , %.3f, %.3f, %.3f \n" % (step, loss, loss_STT,loss_TTS)
                f.write(string_out)
            step += 1

        # now to test the model...
        test_loss = 0.0
        test_STT = 0.0
        test_TTS = 0.0
        counter = 0.0
        for pair in test_stuff:
            path_txt = os.path.join(data_dir.replace('preprocessed',''),pair[0].replace('./',''))
            path_wav = os.path.join(data_dir.replace('preprocessed',''),pair[1].replace('./',''))
            txt_target = np.load(path_txt)['arr_0'].tolist()
            wav_source = np.load(path_wav)['arr_0'].tolist()
            wav_source = librosa.util.normalize(wav_source)
            #wav_source = std_wav.transform(wav_source)
            # split the data into batches
            tmp = batch_split(txt_target,batch_size,
                    is_txt=True,
                    dict_val=dict_txt)
            
            txt_target = torch.FloatTensor(tmp).to(device)
            txt_target = text_pruner(txt_target,length_txt)
            
            wav_source = torch.FloatTensor(batch_split(wav_source,batch_size)).to(device)
            
            # obtain the outputs
            encoded_wav = encoder(wav_source)
            encoded_txt = text_encoder(txt_target).detach()
            latent_wav = latent_1(encoded_wav)
            latent_txt = latent_2(encoded_txt).detach()
            decoded_wav, loss_STT = decoder(latent_wav.to(device),
                                                latent_txt,txt_target)
            decoded_wav, loss_TTS = text_decoder(latent_txt.to(device),
                                                latent_wav,wav_source)
            loss_1 = loss_TTS.item()
            loss_2 = loss_STT.item()
            test_TTS += loss_1
            test_STT += loss_2

            test_loss = test_loss + loss_1+loss_2
            counter += 1.0
        test_loss = test_loss / counter
        test_TTS = test_TTS / counter
        test_STT = test_STT / counter
        print("The number of discarded data is: ",num_discarded)
        test_report = "The validation loss is : total=%.5f, STT=%.5f, TTS=%.5f" %(test_loss, test_STT, test_TTS)
        print(test_report)
        # to save the model...
        if e%1 == 0:
            if not os.path.exists("./models"):
                os.mkdir("./models")
            path = "./models/epoch_%i.pt" %(e)
            torch.save({
                'epoch' : e,
                'encoder_state' : encoder.state_dict(),
                'text_encoder_state': text_encoder.state_dict(),
                'decoder_state' : decoder.state_dict(),
                'text_decoder_state': text_decoder.state_dict(),
                'latent_wav_state' : latent_1.state_dict(),
                'latent_txt_state' : latent_2.state_dict(),
                #'optimizer_state_encoder': opt_1.state_dict(),
                #'optimizer_state_attention': opt_2.state_dict(),
                #'optimizer_state_decoder': opt_d.state_dict(),
                'loss' : test_loss}, path)
    f.close()

if __name__=='__main__':
    trainer()
