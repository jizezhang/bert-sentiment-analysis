#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from collections import namedtuple
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
### YOUR CODE HERE for part 1d
class Highway(nn.Module):
    '''
    implement the highway network:
    - forward
    - 
    '''
    def __init__(self, embedding_size, dropout_rate):
        '''Init the Highway model'''
        super(Highway, self).__init__()
        self.embedding_size = embedding_size
        self.h_projection = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.h_gate = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    
    def forward(self, x_conv_out):
        x_linear1 = self.h_projection(x_conv_out)
        x_linear2 = self.h_gate(x_conv_out)
        x_proj = F.relu(x_linear1)
        x_gate = torch.sigmoid(x_linear2)
        # x_highway = x_gate * x_proj + (torch.ones(self.embedding_size) - x_gate) * x_conv_out
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        x_word_emb = self.dropout(x_highway)
        return x_word_emb