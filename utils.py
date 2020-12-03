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
        loss1 = nn.L1Loss()
        l1 = loss1(x,y)
        loss2 = nn.CosineSimilarity(dim=0, eps=1e-6)
        l2 = loss2(x,y).to(device)
        ones = torch.ones(l2.size()).to(device)
        loss3 = nn.BCEWithLogitsLoss()
        l3 = loss3(x,y)
        return l3 + torch.log(l1)#+ (ones+l2).mean()
    else:
        loss = nn.CosineSimilarity(dim=0, eps=1e-6)
        l = loss(x,y).to(device)
        ones = torch.ones(l.size()).to(device)
        loss2 = nn.MSELoss()
        l2 = loss2(x,y)
        return torch.log(l2) + (-1)*torch.log((ones+l).mean())

def Opt(model,adam=True,learning_rate=0.01):
    if adam:
        return optim.Adam(model.parameters(),lr=learning_rate)
    else:
        return optim.SGD(model.parameters(),lr=learning_rate)

def Schedule(opt,step_size=10,gamma=0.5):
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)
    return scheduler

# ======================================
#
#  ENCODER MODULE
#
# ======================================
class Encoder_Conv(nn.Module):
    # convolution-based encoder module 
    def __init__(self,device,input_dim = 2982, output_dim=62,depth=6):
        super(Encoder_Conv, self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.init_fc = nn.Linear(input_dim, int(round(input_dim/10))).to(device)
        self.rels = nn.LeakyReLU(0.4).to(device)
        self.device = device
        # convolution layers
        conv0 = nn.Conv2d(1,1,3,padding=1,stride=1)
        conv1 = nn.Conv2d(1,4,3,padding=1,stride=2)
        conv2 = nn.Conv2d(4,8,3,padding=1,stride=2)
        conv3 = nn.Conv2d(8,16,3,padding=1,stride=2)
        conv4 = nn.Conv2d(16,32,3,padding=1,stride=2)
        conv5 = nn.Conv2d(32,64,3,padding=1,stride=2)
        self.list_convs = [conv0,conv1, conv2, conv3, conv4, conv5]
        # batch normalizations
        bn0 = nn.InstanceNorm2d(1)
        bn1 = nn.InstanceNorm2d(4)
        bn2 = nn.InstanceNorm2d(8)
        bn3 = nn.InstanceNorm2d(16)
        bn4 = nn.InstanceNorm2d(32)
        bn5 = nn.InstanceNorm2d(64)
        self.list_bns = [bn0, bn1, bn2, bn3, bn4, bn5]
        # final output layer
        self.final_fc = nn.Linear(20,output_dim).to(device) # need to change this
        self.sigs = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        x = self.rels(self.init_fc(x))
        x = x.view(x.size()[0],1,64,298).to(self.device)#int(round(self.input_dim))).to(device)
        for i in range(self.depth):
            funct1 = self.list_convs[i].to(self.device)
            funct2 = self.list_bns[i].to(self.device)
            x = funct1(x)
            x = funct2(x)
        x = x.reshape(2*64,-1)
        x = self.final_fc(x)
        x = self.rels(x.reshape(2,64,-1))
        return x

    def num_flat_features(self,x):
        size = x.size()[1:] # all dims except the batch dim
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# ======================================
#
#   Encoder-to-text module 
#
# ======================================
class ToText(nn.Module):
    def __init__(self,output_size,device):
        super(ToText,self).__init__()
        self.device = device

        conv0 = nn.Conv2d(1,4,3,padding=1, stride=1).to(device)
        conv1 = nn.Conv2d(4,16,3,padding=1, stride=1).to(device)
        conv2 = nn.Conv2d(16,32,3,padding=1, stride=1).to(device)
        conv3 = nn.Conv2d(32,64,3, padding=1, stride=1).to(device)
        conv4 = nn.Conv2d(64,128,3,padding=1,stride=1).to(device)
        bn0   = nn.InstanceNorm2d(4).to(device)
        bn1   = nn.InstanceNorm2d(16).to(device)
        bn2   = nn.InstanceNorm2d(32).to(device)
        bn3   = nn.InstanceNorm2d(64).to(device)
        bn4   = nn.InstanceNorm2d(128).to(device)
        self.convs = [conv0, conv1,conv2,conv3,conv4]
        self.bns = [bn0, bn1, bn2, bn3, bn4]

        self.depth = len(self.convs)

        self.fs_3 = nn.Linear(7936,6000).to(device)
        self.fs_2 = nn.Linear(6000,5000).to(device)
        self.fs_1 = nn.Linear(5000, 4000).to(device)
        self.fs = nn.Linear(4000,output_size).to(device)
        self.rels = nn.LeakyReLU(0.3)
        self.m = nn.Dropout(p=0.2)
    def forward(self,x):
        x = x.reshape(2,1,64,-1)
        for i in range(self.depth):
            f1 = self.convs[i]
            f2 = self.bns[i]
            x = f1(x)
            x = f2(x)
        x = x.reshape(2,64,-1)
        x = self.m(self.rels(self.fs_3(x)))
        x = self.m(self.rels(self.fs_2(x)))
        x = self.m(self.rels(self.fs_1(x)))
        x = self.rels(self.fs(x))

        return x

# ======================================
#
#  Decoder Module
#
# ======================================
class Decoder(nn.Module):
    def __init__(self, output_size, device, batch_size):
        super(Decoder,self).__init__()
        self.bn = batch_size
        # convolution layers
        conv1 = nn.Conv2d(1,4,3,padding=1,stride=1)
        conv2 = nn.Conv2d(4,8,3,padding=1,stride=1)
        conv3 = nn.Conv2d(8,16,3,padding=1,stride=1)
        conv4 = nn.Conv2d(16,32,3,padding=1,stride=1)
        conv5 = nn.Conv2d(32,64,3,padding=1,stride=1)
        self.list_convs = [conv1, conv2, conv3, conv4, conv5]
        # batch normalizations
        bn1 = nn.InstanceNorm2d(4)
        bn2 = nn.InstanceNorm2d(8)
        bn3 = nn.InstanceNorm2d(16)
        bn4 = nn.InstanceNorm2d(32)
        bn5 = nn.InstanceNorm2d(64)
        self.list_bns = [bn1, bn2, bn3, bn4, bn5]

        self.depth = len(self.list_bns)

        self.m = nn.Dropout(p=0.2)
        self.rels = nn.LeakyReLU(0.3)
        self.fs_4 = nn.Linear(3968, 3700).to(device)
        self.fs_3 = nn.Linear(3700,3500).to(device)
        self.fs_2 = nn.Linear(3500,3300).to(device)
        self.fs_1 = nn.Linear(3300,3100).to(device)
        self.fs = nn.Linear(3100,output_size).to(device)
        self.device = device

    def forward(self, x):
        x = x.reshape(self.bn, 1, -1, 62)
        for i in range(self.depth):
            funct1 = self.list_convs[i].to(self.device)
            funct2 = self.list_bns[i].to(self.device)
            x = funct1(x)
            x = self.rels(funct2(x))
        x = x.reshape(2,64,-1)
        x = self.m(self.rels(self.fs_4(x)))
        x = self.m(self.rels(self.fs_3(x)))
        x = self.m(self.rels(self.fs_2(x)))
        x = self.m(self.rels(self.fs_1(x)))
        x = self.fs(x)
        return x
