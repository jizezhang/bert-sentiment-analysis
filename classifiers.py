from cnn import CNN 
import torch.nn as nn
import torch
import torch.optim as optim
from constants import NUM_CAT

class CNNClassifier(nn.Module):

    def __init__(self, embed_layer, embed_dim, kernel_size, num_filter, num_cat=NUM_CAT):
        super(CNNClassifier, self).__init__()
        self.embed_layer = embed_layer
        self.cnn = CNN(embed_dim, kernel_size, num_filter)
        
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        self.linear = nn.Linear(in_features=num_filter, out_features=num_cat)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        """
        :param x: a tuple, (tensor of shape (batch_size, sent_len, embed_dim), tensor of shape (batch_size, sent_len, 1))
        :param y: label
        """
        embedded = self.embed_layer(x)
        return self.log_softmax(self.linear(self.cnn(embedded)))
