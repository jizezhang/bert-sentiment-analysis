from cnn import CNN 
import torch.nn as nn
import torch.nn.functional as F
from constants import NUM_CAT

class CNNClassifier(nn.Module):

    def __init__(self, embed_dim, kernel_size, num_filter):
        super(CNNClassifier, self).__init__()
        self.cnn = CNN(embed_dim, kernel_size, num_filter)
        self.linear = nn.Linear(in_features=num_filter, out_features=NUM_CAT)
        
    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, sent_len, embed_dim)
        """
        return F.softmax(self.linear(self.cnn(x)))
