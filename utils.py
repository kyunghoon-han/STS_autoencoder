import os, csv, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======================================
#
#  PREPROCESSED DATA LOADER
#
# ======================================

def data_list(path='./preprocessing_unit/preprocessed'):
    # this will output a list of txt and wav npz filename pairs
    # this will also output the path
    the_txts = []
    the_wavs = []
    for filename in os.listdir(path):
        if 'prep_data_txt' in filename:
            the_txts.append(filename)
        elif 'prep_data_wav' in filename:
            the_wavs.append(filename)
    output_list = []
    for filename in the_wavs:
        num = get_filenum(filename)
        for txt_file in the_txts:
            if num == get_filenum(txt_file):
                output_list.append([filename,txt_file])
    # outputs the directory and the filenames in it
    return path, output_list

def get_filenum(filename):
    filename = filename.replace('.npz','')
    filename = filename.replace('prep_data_','')
    if 'txt' in filename:
        filename = filename.replace('txt','')
    elif 'wav' in filename:
        filename = filename.replace('wav','')
    return filename

def read_encoded_data(path,filename_pair):
    # get the paths for the text and wav files
    info_txt = os.path.join(path,filename_pair[0])
    info_wav = os.path.join(path,filename_pair[1])
    # read txt
    text = np.load(info_txt)
    wav = np.load(info_txt)
    return text.tolist(), wav.tolist()

# ======================================
#
#  Loss, Optimizer and Scheduler
#
# ======================================
def Loss(x,y):
    funct = nn.BCELoss()
    return funct(x,y)

def Opt(model,learning_rate=0.01):
    return optim.Adam(model.parameters(),lr=learning_rate)

def Schedule(opt,step_size=20,gamma=0.5):
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

# ======================================
#
#  ENCODER MODULE
#
# ======================================
class Encoder_Conv(nn.Module):
    # convolution-based encoder module 
    def __init__(self,device,input_dim = 3977, output_dim=1,depth=4):
        super(Encoder_Conv, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.init_fc = nn.Linear(input_dim, int(round(input_dim/10)))
        self.rels = nn.LeakyReLU(0.4)
        self.device = device
        # convolution layers
        conv1 = nn.Conv2d(1,4,3,padding=1,stride=2)
        conv2 = nn.Conv2d(4,8,3,padding=1,stride=2)
        conv3 = nn.Conv2d(8,16,3,padding=1,stride=2)
        conv4 = nn.Conv2d(16,32,3,padding=1,stride=2)
        self.list_convs = [conv1, conv2, conv3, conv4]
        # batch normalizations
        bn1 = nn.BatchNorm2d(4)
        bn2 = nn.BatchNorm2d(8)
        bn3 = nn.BatchNorm2d(16)
        bn4 = nn.BatchNorm2d(32)
        self.list_bns = [bn1, bn2, bn3, bn4]
        # final output layer
        self.final_fc = nn.Linear(52,output_dim) # need to change this
        self.sigs = nn.Sigmoid()

    def forward(self, x):
        x = self.rels(self.init_fc(x))
        x = x.view(x.size()[0],1,-1,199).to(self.device)#int(round(self.input_dim))).to(device)
        for i in range(self.depth):
            funct1 = self.list_convs[i].to(self.device)
            funct2 = self.list_bns[i].to(self.device)
            x = funct1(x)
            x = funct2(x)
        print(x.size())
        exit()
        x = self.final_fc(x)
        x = x.view(2,64)
        x = self.sigs(x)
        return x.to(self.device)

    def num_flat_features(self,x):
        size = x.size()[1:] # all dims except the batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# ======================================
#
#   GRU-based Encoder 
#
# ======================================
class Encoder_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, device, 
                input_dim=3977,output_dim=62, depth=4,
                batch_size=64, n_layers=3, drop_prob = 0):
        super(Encoder_GRU, self).__init__()
        self.hs = hidden_size
        self.nl = n_layers
        self.bs = batch_size
        self.device=device
        self.depth = depth
        
        self.final_fc1 = nn.Linear(199,62).to(device)
        self.final_fc2 = nn.Linear(4,1).to(device)

        self.init_fc = nn.Linear(input_dim, int(round(input_dim/10))).to(device)
        self.rels = nn.LeakyReLU(0.4)
        self.sigs = nn.Sigmoid()

        # convolution layers
        conv1 = nn.Conv2d(1,4,3,padding=1,stride=2)
        conv2 = nn.Conv2d(4,8,3,padding=1,stride=2)
        conv3 = nn.Conv2d(8,16,3,padding=1,stride=2)
        conv4 = nn.Conv2d(16,32,3,padding=1,stride=2)
        self.list_convs = [conv1, conv2, conv3, conv4]
        # batch normalizations
        bn1 = nn.BatchNorm2d(4)
        bn2 = nn.BatchNorm2d(8)
        bn3 = nn.BatchNorm2d(16)
        bn4 = nn.BatchNorm2d(32)
        self.list_bns = [bn1, bn2, bn3, bn4]

        self.gru = nn.GRU(199, 199, n_layers,
                            dropout=drop_prob,batch_first=True).to(device)
        
        
    def init_hidden(self,hidden_size, batch_size):
        return torch.zeros(self.nl, batch_size, hidden_size, 
                            device=self.device,dtype=torch.float)

    def forward(self, x):
        x = self.rels(self.init_fc(x)).to(self.device)
        x = x.view(x.size()[0],1,-1,199).to(self.device)#int(round(self.input_dim))).to(device)
        #for i in range(self.depth):
        #    funct1 = self.list_convs[i].to(self.device)
        #    funct2 = self.list_bns[i].to(self.device)
        #    x = funct1(x)
        #    x = funct2(x)
        #x = x.float() # okay, I'm inputting a float tensor, but somehow it becomes a long...
                      # hence... this bs has to be added...
        x = x.view(x.size()[0],-1,x.size()[-1])
        hidden = self.init_hidden(hidden_size=x.size()[-1],
                                  batch_size=x.size()[0])
        # pass the embedded input vectors into an LSTM and return all outputs
        x, hidden = self.gru(x,hidden)
        x = x.reshape(-1,199)
        x = self.sigs(self.final_fc1(x))
        x = x.reshape(-1,128,62)

        return x, hidden

# ======================================
#
#    Attention Decoder
#
# ======================================
class AttentionDecoder(nn.Module):
    def __init__(self,hidden_size,output_size, vocab_size,device):
        super(AttentionDecoder,self).__init__()
        self.hs = hidden_size
        self.os = output_size
        self.device = device

        self.gru = nn.GRU(hidden_size + vocab_size, output_size).to(device)
        self.attn = nn.Linear(hidden_size + output_size, 1).to(device)
        self.fc = nn.Linear(output_size, vocab_size).to(device)
        self.final = nn.Linear(129,128).to(device)

        self.sigs = nn.Sigmoid()
        self.rels = nn.LeakyReLU(0.3)

    def init_hidden(self):
        return torch.zeros(1,128,self.os, device=self.device)
    
    def forward(self, prev_hidden, encoder_outputs):
        weights = []
        decoder_hidden = self.init_hidden()
        
        for i in range(len(encoder_outputs)):
            concat = torch.cat((decoder_hidden[0],
                                encoder_outputs[i]),dim=1)
            weights.append(self.attn(concat))
        normalized_weights = F.softmax(torch.cat(weights, 1),1)
        normalized_weights = normalized_weights.unsqueeze(1).reshape(2,1,-1)
        attn_applied = torch.bmm(normalized_weights,encoder_outputs)
        input_gru = torch.cat((attn_applied,
                               encoder_outputs),dim=1)
        input_gru = input_gru.reshape(1,129,-1)
        decoder_hidden = torch.cat((decoder_hidden, 
                                    torch.rand(1,1,decoder_hidden.size()[-1]).to(self.device)),
                                    dim=1)
        output, hidden = self.gru(input_gru, decoder_hidden)
        output = self.rels(self.fc(output))
        output = self.final(torch.transpose(output,1,2))
        output = self.sigs(torch.transpose(output,1,2)).reshape(2,64,-1)

        return output, hidden, normalized_weights

# ======================================
#
#   Score function 
#
# ======================================


