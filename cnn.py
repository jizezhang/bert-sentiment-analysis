# -*- coding: utf-8 -*-
import sys
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.utils
from torch.functional import F

class CNN(nn.Module):
    '''
    implement the cnn
    '''
    def __init__(self, embed_dim, kernel_size, num_filter):
        '''Init the Highway model'''
        super(CNN, self).__init__()
        
        self.kernel_size = kernel_size
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim, 
            out_channels=num_filter, 
            kernel_size=self.kernel_size, 
            bias=True,
        )
    
    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, sent_len, embed_dim)
        """
        sent_len = x.shape[1]
        x_conv = self.conv1(x.permute(0, 2, 1))
        x_relu = F.relu(x_conv)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool1d
        x_maxpool = F.max_pool1d(x_relu, sent_len - self.kernel_size + 1)
        return x_maxpool.squeeze()