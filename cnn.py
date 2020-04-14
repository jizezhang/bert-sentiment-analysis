#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils

class CNN(nn.Module):
    '''
    implement the cnn
    '''
    def __init__(self, embed_size, kernel_size, num_filter):
        '''Init the Highway model'''
        super(CNN, self).__init__()
        
        self.in_channel = embed_size 
        self.kernel_size = kernel_size
        self.out_channel = num_filter
        self.conv1 = nn.Conv1d(self.in_channel, self.out_channel, self.kernel_size, bias=True)
    #(12, 103, 768)
    #(12, 103 - kernel_size + 1, num_filter)
    # (12, 1, num_filter)
    def forward(self, x):
        # x_reshape = x.permute(0,1,3,2)
        max_word_length = x.shape[-1]
        x_conv = self.conv1(x)
        x_relu = F.relu(x_conv)
        x_conv_out = F.max_pool1d(x_relu, max_word_length-self.kernel_size+1)
        return x_conv_out
        
