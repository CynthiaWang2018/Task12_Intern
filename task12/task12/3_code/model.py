# This file is to construct model

import torch.nn as nn
import torch.nn.functional as F
import torch

class Test1Model(nn.Module):
    def __init__(self, device, input_size=12, hidden_size=[128, 256], num_classes=12, num_layer=5):
        super(Test1Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.para_size = input_size

        # 一层卷积神经网络
        self.conv1 = nn.Conv1d(1, self.hidden_size[0], kernel_size=9, padding=4)
        # in_channels, out_channels, kernel_size   (9-1)/2 = 4 为了不变长度, 设置padding的大小为4

        # 第二层卷积神经网络
        self.conv2 = nn.Conv1d(self.hidden_size[0], self.hidden_size[0], kernel_size=5, padding=2)
        # in_channels, out_channels, kernel_size   (5-1)/2 = 2 为了不变长度, 设置padding的大小为2

        # 第三层LSTM
        self.rnn = nn.LSTM(
            input_size = self.hidden_size[0], # dimVectors 128
            hidden_size = self.hidden_size[1], # hidden size 256
            num_layers = self.num_layer, # num of hidden layer 3
            batch_first = True, # input & output will has batch size as 1s dimension. batch_size 作为第一维度[32, 1, 128]...
        )
        # input_size, hidden_dim, num_layers

        self.h = torch.rand(num_layer, 32, self.hidden_size[1]).to(device)  # this is important
        # num_layer, batch_size, hidden_dim
        self.out = nn.Linear(self.hidden_size[1], self.num_classes)

        self.linear1 = nn.Linear(12 * self.hidden_size[1], self.hidden_size[1])
        self.linear2 = nn.Linear(self.hidden_size[1], self.num_classes)
        # self.Sigmoid = nn.Sigmoid()

    def forward(self, x): # [32, 12]

        ###print()
        #print('x1', x)
        # print('x1.shape', x.shape)
        x = x.unsqueeze(1) # [32 12] -> [32, 1, 12]
        ###print()
        #print('x2', x)
        # print('x2.shape', x.shape)

        h = self.h # [3, 32, 256]
        ###print()
        # print('h.shape', h.shape)

        x = self.conv1(x) # [32, 32, 12]
        ###print()
        # print('x3', x.shape)
        x = self.conv2(x) # [32, 32, 12]
        ###print()
        # print('x4', x.shape)
        x = x.permute(0, 2, 1) # [32, 12, 32]
        ###print()
        # print('x5', x.shape)
        r_out, (h_n, h_c) = self.rnn(x, (h, h))  # r_out [32, 12, 256] h_n [3, 32, 356] h_c [3, 32, 256]
        ###print()
        # print('r_out', r_out.shape)
        # print('r_out', r_out)
        # print('h_n', h_n.shape)
        # print('h_c', h_c.shape)

        r_out = r_out.contiguous() # [32, 12, 256]
        ###print()
        # print('r_out2', r_out.shape)

        r_out = r_out.view(32, -1)  # [32, 3072]
        ###print()
        # print('r_out3', r_out.shape)

        out = self.linear1(r_out)  # [32, 256]
        ###print()
        # print('out', out.shape)
        #out = F.relu(out)
        out2 = self.linear2(out) # [32, 12]
        ###print()
        # print('out2', out2.shape)
        return out2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Test1Model(device, input_size=12, hidden_size=[32, 128], num_classes=12, num_layer=3).to(device)
# print(model)