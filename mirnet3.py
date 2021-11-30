import argparse
import os
import random
import shutil
import time
import warnings
import torch.nn as nn
import torch
import math

class FourierLayer(nn.Module):
    
    def __init__(self, size):
        super().__init__()
        self.size_in, self.size_out = size, size
        weight = torch.Tensor(1)
        self.weight = nn.Parameter(weight)  
        bias = torch.Tensor(1)
        self.bias = nn.Parameter(bias)
        self.range = torch.tensor(range(1, size+1))
        #make sure the values are between 0 and 1

        # initialize weights and biases
        nn.init.uniform_(self.weight, a=0, b=1.0)
        nn.init.uniform_(self.bias, a=0, b=1.0)

    def forward(self, x):
    	#adjusted = torch.tensor([])
    	return torch.mul(x, torch.cos(torch.mul(self.weight, self.range).add(self.bias)))

class FourierLayer2(nn.Module):
    
    def __init__(self, size, size2):
        super().__init__()
        self.frame_size, self.wave_size = size, size2
        weight = torch.Tensor((self.frame_size, self.wave_size))
        self.weight = nn.Parameter(weight)  
        bias = torch.Tensor((self.frame_size, self.wave_size))
        self.bias = nn.Parameter(bias)
        ran = []
        for i in range(self.frame_size):
            ran.append(range(1, self.wave_size+1))

        self.range = torch.tensor(ran)
        #make sure the values are between 0 and 1

        # initialize weights and biases
        nn.init.uniform_(self.weight, a=0, b=1.0)
        nn.init.uniform_(self.bias, a=0, b=1.0)

    def forward(self, x):
        #adjusted = torch.tensor([])
        return torch.mul(x, torch.cos(torch.mul(self.weight, self.range).add(self.bias)))

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
        nn.init.uniform_(self.freqs, a=0, b=1.0)
        nn.init.uniform_(self.phases, a=0, b=1.0)

    def forward(self, x):
        fourier_size = (x.shape[-1], self.num_freqs)
        #generate basis of cosine waves
        cos_waves = torch.empty(fourier_size,dtype=torch.float)
        for i in range(self.num_freqs):
            cos_waves[:,i] = torch.cos(torch.mul(self.freqs[i], torch.arange(0,self.wave_size)).add(self.phases[i]))

        #perform fourier transform via matrix multiplication and mean operation
        x = torch.unsqueeze(x,-2)
        freq_coeffs = torch.mul(1/self.wave_size, torch.matmul(x,cos_waves) )
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
    def __init__(self, num_class=722, seq_len=31, leaky_relu_slope=0.01):
        super().__init__()
        self.seq_len = seq_len  # 31
        self.num_class = num_class
        #self.fouriers = []
        self.num_freqs = 513
        self.fourier = FourierLayer3(1024, self.num_freqs)
        #for i in range(self.num_freqs):
            #self.fouriers.append(FourierLayer2(31, 1024))
            #self.fouriers.append(FourierLayer3(1024, self.num_freqs))

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
        #z_size = x.shape[:-1] + (self.num_freqs,)
        #z = torch.empty(z_size,dtype=torch.float)
        #for i in range(self.num_freqs):
        #    z[:,:,:,i] = self.fouriers[i](x)
        z = self.fourier(x)

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

class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
    def __init__(self, num_classes=722, blur_range=3):
        super().__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range
        self.gaussian_decays = [self.gaussian_val(dist=d) for d in range(blur_range + 1)]

    @staticmethod
    def gaussian_val(dist: int, sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        with torch.no_grad():
            pred_logit = torch.log_softmax(pred, dim=self.dim)
            target_onehot = empty_onehot(target, self.num_classes).to(target.device)
            for dist in range(self.blur_range, -1, 1):
                for direction in [1, -1]:
                    blur = torch.clamp(target + (direction * dist), min=0, max=self.num_classes - 1)
                    target_onehot = target_onehot.scatter_(dim=2, index=torch.unsqueeze(blur, dim=2), value=self.gaussian_decays[dist])
            target_loss_sum = target_onehot-(pred_logit *t).sum(dim=self.dim) #*t?
            return target_loss_sum.mean()

class CrossEntropyLossWithGaussianSmoothedLabels2(nn.Module):
    def __init__(self, num_classes=722, blur_range=3):
        super().__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range
        self.gaussian_decays = [self.gaussian_val(dist=d) for d in range(blur_range + 1)]

        self.cross_entropy = torch.nn.CrossEntropyLoss()

    @staticmethod
    def gaussian_val(dist: int, sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):

        with torch.no_grad():
            pred_logit = torch.log_softmax(pred, dim=self.dim)
            target_onehot = empty_onehot(target, self.num_classes).to(target.device)
            for dist in range(self.blur_range, -1, 1):
                for direction in [1, -1]:
                    blur = torch.clamp(target + (direction * dist), min=0, max=self.num_classes - 1)
                    target_onehot = target_onehot.scatter_(dim=2, index=torch.unsqueeze(blur, dim=2), value=self.gaussian_decays[dist])

            pred_logit = pred_logit.view(pred_logit.shape[0]*pred_logit.shape[1],pred_logit.shape[2])
            return self.cross_entropy(pred_logit, target_onehot)

#net = Mirnet()
#loss_fn = CrossEntropyLossWithGaussianSmoothedLabels()
