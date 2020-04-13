#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, kenerl_size=5):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(CNNClassifier, self).__init__()
        self.embed_size = embed_size

        self.kenerl_size = kenerl_size
        self.embeddings = nn.Embedding(len(vocab.char2id), self.embed_char, padding_idx=pad_token_idx)
        self.cnn = CNN(self.embed_char, self.kenerl_size, self.embed_size)#self.embed_size is number of filter
        self.highway = Highway(self.embed_size, dropout_rate=0.3)


    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1f
        #print(input.shape)
        output = self.embeddings(input) 
        #print(output.shape)
        x_reshape = output.permute(0,1,3,2) ##(sentence_length, batch_size, ebed_char, max_word_length)
        shape = x_reshape.shape
        sentence_length = shape[0]
        batch_size = shape[1]
        max_word_length = shape[3]
        x_reshape = x_reshape.view(-1, self.embed_char, max_word_length)
        #print(x_reshape.shape)

        x_cnn = self.cnn.forward(x_reshape)
        #print(x_cnn.shape)
        x_highway = self.highway.forward(x_cnn.view(sentence_length, batch_size, self.embed_size))

        return x_highway
# (sentence_length, batch_size, embed_size, max_word_length-k+1)
# batch_size, sentence_length, embed_size, max_word_length-k+1
        ### END YOUR CODE