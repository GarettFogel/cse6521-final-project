import argparse
import os
import random
import shutil
import time
import warnings
import torch.nn as nn

parser = argparse.ArgumentParser(description='final project')
#convblock -- torch.nn.Conv2d(), torch.nn.BatchNorm2D, torch nn LeakyRelu, torch.nn.conv2D

class FourierLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, size):
        super().__init__()
        self.size_in, self.size_out = size, size
        weights = torch.Tensor(size)
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size)
        self.bias = nn.Parameter(bias)
        self.range = torch.tensor(range(1, size+1))

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5)) # weight init
        #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        #bound = 1 / math.sqrt(fan_in)
        #nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
    	adjusted = torch.tensor([])
    	return torch.mul(x, torch.cos(torch.mul(self.weights, self.range).add(self.bias)))

    	#for i, x in enumerate(self.weights):
    		#val = torch.tensor([np.cos(self.weights[i][0] * i + self.weights[i][1])])
    		#torch.cat((adjusted,val.view(1)))        
        #return torch.mul(x, adjusted) 




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
# input layer
#conv block
# resnet block (three of them!)
#pool block (?)
#bi-lstm
#nn.linear(722, 31 * 722)
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs), nn.BatchNorm2d(out_channels))
class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic ResNet block composed by two layers of 3x3conv/batchnorm/activation
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activation_func(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False),
        )

class BiLSTM(nn.Module):
    
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.hidden_size = 64
        drp = 0.1
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(64, 1)


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        #print("avg_pool", avg_pool.size())
        #print("max_pool", max_pool.size())
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out

def main():
	args = parser.parse_args()
