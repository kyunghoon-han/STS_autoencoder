import os, csv, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ======================================
#
#  Loss, Optimizer and Scheduler
#
# ======================================
def Criterion(x,y,device,switch=False):
    if not switch:
        #loss1 = nn.MSELoss()
        #l1 = loss1(x,y)
        loss2 = nn.BCEWithLogitsLoss()
        l2 = loss2(x,y).to(device)
        return l2
    else:
        #loss = nn.BCEWithLogitsLoss()
        #l = loss(x,y).to(device)
        loss2 = nn.L1Loss()
        l2 = loss2(x,y)
        return l2 #+ l

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
#  TEXT-ENCODER MODULE
#
# ======================================
class TEncoder(nn.Module):
    # RNN-based text encoder module
    def __init__(self,device,input_dim = 62, output_dim=34, hidden_size=10,num_layers=5):
        super(TEncoder, self).__init__()
        self.input_dim = input_dim
        self.init_fc = nn.Linear(input_dim, 31).to(device)
        self.rels = nn.LeakyReLU(0.4).to(device)
        self.device = device
        # sprinkle of a RNN
        self.rnn = nn.RNN(input_size=31, hidden_size=hidden_size,num_layers=num_layers, dropout=0.3).to(device)
        self.hidden_size=hidden_size
        self.num_layers = num_layers
        # final output layer
        self.final_2 = nn.Linear(170, 90).to(device)
        self.final_1 = nn.Linear(90,45).to(device)
        self.final_fc = nn.Linear(45,output_dim).to(device) # need to change this
        self.sigs = nn.Sigmoid()
    def forward(self, x):
        x = x.to(self.device)
        x = self.rels(self.init_fc(x))
        hidden = torch.zeros(self.num_layers,4,self.hidden_size).to(self.device)
        x, hidden = self.rnn(x,hidden)
        x = x.reshape(2,4,-1)
        x = self.rels(self.final_2(x))
        x = self.rels(self.final_1(x))
        x = self.final_fc(x)
        return x

# ======================================
#
#   Intermediate latent network 
#
# ======================================
class Latent(nn.Module):
    def __init__(self, input_size,device):
        super(Latent,self).__init__()
        self.rels = nn.LeakyReLU(0.3)
        self.f1 = nn.Linear(input_size,input_size*2).to(device)
        self.f4 = nn.Linear(input_size*2, input_size).to(device)

    def forward(self,x):
        x = self.rels(self.f1(x))
        x = self.rels(self.f4(x))
        return x

# ======================================
#
#   Bahdanau Attention
#
# ======================================
class AdditiveAttention(nn.Module):
    def __init__(self, d_model):
        super(AdditiveAttention, self).__init__()
        self.query_proj = Linear(d_model, d_model, bias=False)
        self.key_proj = Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.rand(d_model).uniform_(-0.1, 0.1))
        self.score_proj = Linear(d_model, 1)

    def forward(self, query, key, value):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)

        context += query

        return context, attn

# ==========================================
#
#  ClovaAI Attention Algorithm
#   - from: KoSpeech GitHub
#
# ==========================================
class LocationAwareAttention(nn.Module):
    def __init__(self, d_model: int = 512, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.d_model = d_model
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=3, padding=1)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.rand(d_model).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(d_model, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query, value, last_alignment_energy):
        batch_size, hidden_dim, seq_len = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_len)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignmment_energy = self.score_proj(torch.tanh(
                self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
                + last_alignment_energy
                + self.bias
        )).squeeze(-1)

        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(-1).unsqueeze(-1))

        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)

        context = torch.bmm(alignmment_energy.unsqueeze(1), value).squeeze(1)

        return context, alignmment_energy

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
#  Text Decoder Module
#
# ======================================
class TDecoder(nn.Module):
    def __init__(self, input_size, batch_size, device, 
                hidden_size=124,num_layers=120):
        super(TDecoder,self).__init__()
        self.bn = batch_size
        self.rels = nn.LeakyReLU(0.2)
        self.device = device
        self.attention = LocationAwareAttention(d_model=input_size).to(device)
        self.fl1 = nn.Linear(34,34*32).to(device)
        self.fl2 = nn.Linear(68,680).to(device)
        self.fl3 = nn.Linear(680,1256).to(device)

    def forward(self, x, encoded_counterpart, target_tensor, backprop=True):
        loss = 0.0
        for a in range(self.bn):
            if a == 0:
                lat = None
            y, lat = self.attention(x, encoded_counterpart,lat)
            y = self.rels(self.fl1(y))
            y = y.reshape(32,-1)
            y = self.rels(self.fl2(y))
            y = self.fl3(y)
            if backprop:
                crit = Criterion(y, target_tensor[:,a,:],self.device,switch=True)
                loss += crit
            if a == 0:
                output_stuff = y
                s = y.size()
                output_stuff = output_stuff.reshape(s[0],1,s[-1])
            else:
                y = y.reshape(y.size()[0],1,y.size()[-1])
                output_stuff = torch.cat((output_stuff, y),1)
        if backprop:
            loss = loss / self.bn

        return output_stuff, loss
# ======================================
#
#  Decoder Module
#
# ======================================
class Decoder(nn.Module):
    def __init__(self, input_size, batch_size, device, 
                hidden_size=124,num_layers=120):
        super(Decoder,self).__init__()
        self.bn = batch_size
        self.rels = nn.LeakyReLU(0.2)
        self.device = device
        self.attention = LocationAwareAttention(d_model=input_size).to(device)
        self.fl1 = nn.Linear(34,34*34).to(device)
        self.fl2 = nn.Linear(68,62).to(device)

    def forward(self, x, encoded_counterpart, target_tensor,backprop=True):
        loss = 0.0
        for a in range(self.bn):
            if a == 0:
                lat = None
            y, lat = self.attention(x, encoded_counterpart,lat)
            y = self.rels(self.fl1(y))
            y = y.reshape(34,-1)
            y = self.fl2(y)
            if backprop:
                crit = Criterion(y, target_tensor[:,a,:],
                            self.device,switch=False)
                loss += crit
            if a == 0:
                output_stuff = y
                s = output_stuff.size()
                output_stuff = output_stuff.reshape(s[0],1,s[-1])
            else:
                y = y.reshape(y.size()[0],1,y.size()[-1])
                output_stuff = torch.cat((output_stuff, y),1)
        if backprop:
            loss = loss / self.bn

        return output_stuff, loss
