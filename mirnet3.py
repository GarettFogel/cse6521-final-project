import argparse
import os
import random
import shutil
import time
import warnings
import torch.nn as nn
import torch
import math

class FourierLayer3(nn.Module):
    
    def __init__(self, wave_size, num_freqs):
        super().__init__()
        self.num_freqs = num_freqs
        self.wave_size = wave_size
        freqs = torch.Tensor((num_freqs))
        self.freqs = nn.Parameter(freqs)  
        phases = torch.Tensor((num_freqs))
        self.phases = nn.Parameter(phases)

        # initialize weights and biases
        #nn.init.uniform_(self.freqs, a=0, b=1.0)
        #nn.init.uniform_(self.phases, a=0, b=1.0)
        samp_rate = 8000
        min_freq = 2*math.pi*27 / samp_rate #A0
        max_freq = 2*math.pi*4186.6 / samp_rate #C8 #1/2 #avoid nyquist 
        nn.init.uniform_(self.freqs, a=min_freq, b=max_freq)
        nn.init.uniform_(self.phases, a=0, b=math.pi)

    def forward(self, x):
        fourier_size = (x.shape[-1], self.num_freqs)
        #generate basis of cosine waves
        cos_waves = torch.empty(fourier_size,dtype=torch.float)
        for i in range(self.num_freqs):
            cos_waves[:,i] = torch.cos(torch.mul(self.freqs[i], torch.arange(0,self.wave_size)).add(self.phases[i]))

        #perform fourier transform via matrix multiplication and mean operation
        x = torch.unsqueeze(x,-2)
        freq_coeffs = torch.abs(torch.mul(1/self.wave_size, torch.matmul(x,cos_waves)))
        freq_coeffs = torch.squeeze(freq_coeffs,-2)
        return freq_coeffs
        #return torch.mul(x, torch.cos(torch.mul(self.freq, torch.arange(0,self.wave_size)).add(self.phase)))

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, leaky_relu_slope=0.01):
        super().__init__()
        self.downsample = in_channels != out_channels

        
        self.pre_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  
        )


        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        )


        self.conv1by1 = None
        if self.downsample:
            self.conv1by1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pre_conv(x)
        if self.downsample:
            x = self.conv(x) + self.conv1by1(x)
        else:
            x = self.conv(x) + x
        return x

class Mirnet(nn.Module):
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01, use_f_layer=True):
        super().__init__()
        self.seq_len = seq_len  # 31
        self.num_class = num_class
        #self.fouriers = []
        self.num_freqs = 513
        if(use_f_layer==True):
            self.fourier = FourierLayer3(1024, self.num_freqs)
        else:
            self.fourier = torch.nn.Identity()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),  
        )
        self.pool_block = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.MaxPool2d(kernel_size=(1, 4)),  
            nn.Dropout(p=0.5),
        )

        self.res_block1 = ResBlock(in_channels=64, out_channels=128)  
        self.res_block2 = ResBlock(in_channels=128, out_channels=192) 
        self.res_block3 = ResBlock(in_channels=192, out_channels=256)  
        
        self.detector_conv = nn.Sequential(
            nn.Conv2d(640, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(p=0.5),
        )


        self.bilstm_classifier = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True) 


        self.bilstm_detector = nn.LSTM(
            input_size=512, hidden_size=256,
            batch_first=True, dropout=0.3, bidirectional=True)  

        self.classifier = nn.Linear(in_features=512, out_features=self.num_class)  

      
        self.detector = nn.Linear(in_features=512, out_features=2)  
        self.apply(self.init_weights)

    def forward(self, x):
        #Apply fourier layer or identify if data was preprocessed with fourier transform
        z = self.fourier(x)

        #Rest of network
        convblock_out = self.conv_block(z)
        resblock1_out = self.res_block1(convblock_out)
        resblock2_out = self.res_block2(resblock1_out)
        resblock3_out = self.res_block3(resblock2_out)
        poolblock_out = self.pool_block(resblock3_out)
        classifier_out = poolblock_out.permute(0, 2, 1, 3).contiguous().view((-1, 31, 512))
        classifier_out, _ = self.bilstm_classifier(classifier_out)  

        classifier_out = classifier_out.contiguous().view((-1, 512))  
        classifier_out = self.classifier(classifier_out)
        classifier_out = classifier_out.view((-1, 31, self.num_class))  

        return classifier_out

    def predict(self,raw_pred):
        pred_logits = torch.softmax(raw_pred, dim=-1)
        preds = torch.argmax(pred_logits,-1)
        return preds.view((-1))

    @staticmethod
    def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
                for p in m.parameters():
                    if p.data is None:
                        continue

                    if len(p.shape) >= 2:
                        nn.init.orthogonal_(p.data)
                    else:
                        nn.init.normal_(p.data)
            else:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.uniform_(m.weight)
                if hasattr(m, 'bias') and  m.bias is not None:
                    nn.init.constant_(m.bias, 0)

def empty_onehot(target: torch.Tensor, num_classes: int):
    #onehot_size = target.size() + (num_classes, )
    onehot_size = (target.shape[0] * target.shape[2], num_classes )
    return torch.FloatTensor(*onehot_size).zero_()


def to_onehot(target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None):
    if src_onehot is None:
        one_hot = empty_onehot(target, num_classes)
    else:
        one_hot = src_onehot

    last_dim = len(one_hot.size()) - 1
    with torch.no_grad():
        one_hot = one_hot.scatter_(
            dim=last_dim, index=torch.unsqueeze(target, dim=last_dim), value=1.0)
    return one_hot

class CrossEntropyLossWithGaussianSmoothedLabels2(nn.Module):
    def __init__(self, num_classes=722, blur_range=3):
        super().__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range
        self.gaussian_decays = [self.gaussian_val(dist=d) for d in range(blur_range + 1)]

        reweight = torch.ones(self.num_classes)
        reweight[-1] = 1/50
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=reweight)

    @staticmethod
    def gaussian_val(dist: int, sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred_logit = torch.softmax(pred, dim=self.dim)
        #reshape tensors to 2d
        pred_logit = pred_logit.view(pred_logit.shape[0]*pred_logit.shape[1],pred_logit.shape[2])
        target = target.view(target.shape[0]*target.shape[2],target.shape[3])

        #return self.cross_entropy(pred_logit, target_onehot)
        return self.cross_entropy(pred_logit, target)

#net = Mirnet()
#loss_fn = CrossEntropyLossWithGaussianSmoothedLabels()
