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
def LossMSE(x,y,device,switch=False):
    if not switch:
        #loss1 = nn.MSELoss()
        #l1 = loss1(x,y)
        loss2 = nn.BCEWithLogitsLoss()
        l2 = loss2(x,y).to(device)
        return l2
    else:
        loss = nn.BCEWithLogitsLoss()
        l = loss(x,y).to(device)
        loss2 = nn.MSELoss()
        l2 = loss2(x,y)
        return l2 + l

def Opt(model,adam=True,learning_rate=0.01):
    if adam:
        return optim.Adam(model.parameters(),
                lr=learning_rate,weight_decay=0.0000001,amsgrad=True)
    else:
        return optim.SGD(model.parameters(),
                lr=learning_rate,momentum=0.0005,nesterov=True)

def Schedule(opt,step_size=500,gamma=0.5):
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    return scheduler

# ======================================
#
#  ENCODER MODULE
#
# ======================================
class Encoder_Conv(nn.Module):
    # convolution-based encoder module 
    def __init__(self,device,input_dim = 1256, output_dim=34, hidden_size=64,num_layers=64):
        super(Encoder_Conv, self).__init__()
        self.input_dim = input_dim
        self.init_fc = nn.Linear(input_dim, int(round(input_dim/10))).to(device)
        self.rels = nn.LeakyReLU(0.4).to(device)
        self.device = device
        # convolution layers
        self.conv0 = nn.Conv2d(1,1,3,padding=1,stride=1).to(device)
        self.conv1 = nn.Conv2d(1,4,3,padding=1,stride=2).to(device)
        self.conv2 = nn.Conv2d(4,8,3,padding=1,stride=2).to(device)
        self.conv3 = nn.Conv2d(8,16,3,padding=1,stride=2).to(device)
        self.conv4 = nn.Conv2d(16,32,3,padding=1,stride=2).to(device)
        self.conv5 = nn.Conv2d(32,64,3,padding=1,stride=2).to(device)
        # batch normalizations
        self.bn0 = nn.InstanceNorm2d(1).to(device)
        self.bn1 = nn.InstanceNorm2d(4).to(device)
        self.bn2 = nn.InstanceNorm2d(8).to(device)
        self.bn3 = nn.InstanceNorm2d(16).to(device)
        self.bn4 = nn.InstanceNorm2d(32).to(device)
        self.bn5 = nn.InstanceNorm2d(64)
        # sprinkle of a RNN
        self.rnn = nn.RNN(input_size=1024, hidden_size=64,num_layers=num_layers, dropout=0.3).to(device)
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        # final output layer
        self.final_fc = nn.Linear(64,output_dim).to(device) # need to change this
        self.sigs = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        x = self.rels(self.init_fc(x))
        x = x.view(x.size()[0],1,4,-1).to(self.device)#int(round(self.input_dim))).to(device)
        x = self.bn0(self.conv0(x))
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = self.bn5(self.conv5(x))
        #x = x.reshape(2*4,-1)
        hidden = torch.zeros(self.num_layers,4,self.hidden_size).to(self.device)
        x = x.reshape(2,4,-1)
        x, hidden = self.rnn(x,hidden)
        x = self.final_fc(x)
        x = self.rels(x.reshape(2,4,-1))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] # all dims except the batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# ======================================
#
#   Intermediate latent network 
#
# ======================================
class Latent(nn.Module):
    def __init__(self, input_size, device):
        super(Latent,self).__init__()
        self.device = device
        self.d = nn.Dropout(p=0.2)
        self.rels = nn.LeakyReLU(0.3)
        self.f1 = nn.Linear(input_size,input_size*2).to(device)
        self.f2 = nn.Linear(input_size*2, input_size*3).to(device)
        self.f3 = nn.Linear(input_size*3, input_size*2).to(device)
        self.f4 = nn.Linear(input_size*2, input_size).to(device)
    def forward(self,x):
        x = self.d(self.rels(self.f1(x)))
        x = self.d(self.rels(self.f2(x)))
        x = self.d(self.rels(self.f3(x)))
        x = self.d(self.rels(self.f4(x)))
        return x


# ======================================
#
#   Encoder-to-text module 
#
# ======================================
class ToText(nn.Module):
    def __init__(self,input_size,output_size,device,hidden_size=64,num_layers=32):
        super(ToText,self).__init__()
        self.device = device

        self.conv0 = nn.Conv2d(1,4,3,padding=1, stride=1).to(device)
        self.conv1 = nn.Conv2d(4,16,3,padding=1, stride=1).to(device)
        self.conv2 = nn.Conv2d(16,32,3,padding=1, stride=1).to(device)
        self.conv3 = nn.Conv2d(32,64,3, padding=1, stride=1).to(device)
        self.conv4 = nn.Conv2d(64,128,3,padding=1,stride=1).to(device)
        self.bn0   = nn.InstanceNorm2d(4).to(device)
        self.bn1   = nn.InstanceNorm2d(16).to(device)
        self.bn2   = nn.InstanceNorm2d(32).to(device)
        self.bn3   = nn.InstanceNorm2d(64).to(device)
        self.bn4   = nn.InstanceNorm2d(128).to(device)

        self.fs_init = nn.Linear(input_size,input_size*2).to(device)
        self.fs_3 = nn.Linear(64*16,17*128).to(device)
        self.fs_2 = nn.Linear(128,64).to(device)
        self.fs_1 = nn.Linear(64, 8).to(device)
        self.fs = nn.Linear(8,output_size).to(device)
        self.rels = nn.LeakyReLU(0.3)
        self.m = nn.Dropout(p=0.2)

        # sprinkle of a RNN
        self.rnn = nn.RNN(input_size=544, hidden_size=64,num_layers=num_layers, dropout=0.3).to(device)
        self.hidden_size=hidden_size
        self.num_layers = num_layers
 

    def forward(self,x):
        x = self.m(self.rels(self.fs_init(x)))
        x = x.reshape(2,1,4,-1)
        x = self.bn0(self.conv0(x))
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.bn4(self.conv4(x))
        x = x.reshape(32,4,-1)
        hidden = torch.zeros(self.num_layers,4,self.hidden_size).to(self.device)
        x, hidden = self.rnn(x,hidden)
        x = x.reshape(2,4,-1)
        x = self.m(self.rels(self.fs_3(x)))
        x = x.reshape(34,4,-1)
        x = self.m(self.rels(self.fs_2(x)))
        x = self.m(self.rels(self.fs_1(x)))
        x = self.fs(x)

        return x

# ======================================
#
#  Decoder Module
#
# ======================================
class Decoder(nn.Module):
    def __init__(self, input_size, output_size, device, 
                batch_size,hidden_size=64,num_layers=120):
        super(Decoder,self).__init__()
        self.bn = batch_size
        # convolution layers
        self.conv1 = nn.Conv2d(1,4,3,padding=1,stride=1).to(device)
        self.conv2 = nn.Conv2d(4,8,3,padding=1,stride=1).to(device)
        self.conv3 = nn.Conv2d(8,16,3,padding=1,stride=1).to(device)
        self.conv4 = nn.Conv2d(16,32,3,padding=1,stride=1).to(device)
        self.conv5 = nn.Conv2d(32,64,3,padding=1,stride=1).to(device)
        # batch normalizations
        self.bn1 = nn.BatchNorm2d(4).to(device)
        self.bn2 = nn.BatchNorm2d(8).to(device)
        self.bn3 = nn.BatchNorm2d(16).to(device)
        self.bn4 = nn.BatchNorm2d(32).to(device)
        self.bn5 = nn.BatchNorm2d(64).to(device)

        self.m = nn.Dropout(p=0.2)
        self.rels = nn.LeakyReLU(0.3)

        self.rnn = nn.RNN(input_size=272, hidden_size=64,num_layers=num_layers, dropout=0.3).to(device)
        self.hidden_size=hidden_size
        self.num_layers = num_layers

        self.fs_init = nn.Linear(input_size,input_size*2).to(device)
        self.fs_4 = nn.Linear(64,128).to(device)
        self.fs_3 = nn.Linear(128,400).to(device)
        self.fs_2 = nn.Linear(400,800).to(device)
        self.fs_1 = nn.Linear(800,1600).to(device)
        self.fs = nn.Linear(1600,output_size).to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.m(self.rels(self.fs_init(x)))
        x = x.reshape(self.bn, 1, -1, 34).to(self.device)
        x = self.rels(self.bn1(self.conv1(x)))
        x = self.rels(self.bn2(self.conv2(x)))
        x = self.rels(self.bn3(self.conv3(x)))
        x = self.rels(self.bn4(self.conv4(x)))
        x = self.rels(self.bn5(self.conv5(x)))
        x = x.reshape(32,4,-1)
        hidden = torch.zeros(self.num_layers,4,self.hidden_size).to(self.device)
        x, hidden = self.rnn(x,hidden)
        x = self.m(self.rels(self.fs_4(x)))
        x = self.m(self.rels(self.fs_3(x)))
        x = self.m(self.rels(self.fs_2(x)))
        x = self.m(self.rels(self.fs_1(x)))
        x = self.fs(x)
        return x
