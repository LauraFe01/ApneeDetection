import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class conv_block(nn.Module):
    def __init__(self, in_chan, out_chan, ker_size, stride, pool_size):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, ker_size, stride, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        out = self.mpool(x)
        return out


class CRNN_2(nn.Module):
    """
    input: [Batch, Channels, Frequency, Time]
    output: [Batch, 1]
    """
    def __init__(self, in_chan=1):
        super(CRNN_2, self).__init__()
        self.conv1 = conv_block(in_chan, 96, ker_size=(5, 5), stride=1, pool_size=(4, 2))
        self.conv2 = conv_block(96, 128, ker_size=(5, 5), stride=1, pool_size=(4, 2))
        self.conv3 = conv_block(128, 128, ker_size=(5, 5), stride=1, pool_size=(2, 2))
        self.conv4 = conv_block(128, 128, ker_size=(3, 3), stride=1, pool_size=(2, 2))
        self.rnn1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True) # serve per analizzare gli spettrogrammi secondo unità di tempo
        self.dense = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        #print(f'After conv1: {x.shape}')
        x = self.conv2(x)
        #print(f'After conv2: {x.shape}')
        x = self.conv3(x) 
        #print(f'After conv3: {x.shape}')
        x = self.conv4(x)
        #print(f'After conv4: {x.shape}')
        x = x.squeeze(2)
        #print(f'After squeeze: {x.shape}')
        x = x.permute(0, 2, 1)
        #print(f'After permute: {x.shape}')
        # RNN
        x, _ = self.rnn1(x)
        # bisogna usare i layer densi e sigmoide precedentemente indicati, quindi
        x = x[:, -1, :]          #[B, 128]
        x = self.dense(x)         #[B,1]
        x = self.sigmoid(x)       #[B,1]
        return x 
