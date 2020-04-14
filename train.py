from embedding import Embeddings
from transformers import *
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils import pad_sents, batch_iter
import math
from cnn import CNN
embed_model = Embeddings()

df_train = pd.read_csv('/Users/ziqunye/Documents/stanford/project/SentimentAnalysis/bert-sentiment-analysis/data/train.csv')
df_test = pd.read_csv('/Users/ziqunye/Documents/stanford/project/SentimentAnalysis/bert-sentiment-analysis/data/test.csv')
training_data_size = 12
df_train = df_train.iloc[:training_data_size, :]
df_test = df_test.iloc[:training_data_size, :]
 # you can experiment with more data to get a more realistic performance score. With a fewer datapoints the model tends to overfit

text_train = df_train['Text'].iloc[0:training_data_size].values
y_train = df_train['Score'].iloc[0:training_data_size].values

text_test = df_train['Text'].iloc[0:training_data_size].values
y_test = df_train['Score'].iloc[0:training_data_size].values


x_train = []
for sentence in tqdm(text_train):
    x_train.append(embed_model.sentence2vec(sentence, layers=2))
    
x_test = []
for sentence in tqdm(text_test):
    x_test.append(embed_model.sentence2vec(sentence, layers=2))

max_length_train = max([len(x) for x in x_train])

x_train = pad_sents(x_train)
x_test = pad_sents(x_test)
# print(len(x_train[0]), len(x_train[0][0]))
# print(len(x_test[0]), len(x_test[0][0]))

x_train = torch.tensor(x_train)
x_test = torch.tensor(x_test)
assert len(x_train) == len(y_train)


class CNNClassifier(nn.Module):
    def __init__(self, embed_size, kernel_size, num_filter):
        super(CNNClassifier, self).__init__()
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.cnn = CNN(embed_size, kernel_size, num_filter)
        self.linear = nn.Linear(in_features=self.num_filter, out_features=1)

    def forward(self, input):
        x_conv_out = self.cnn(input)
        x_conv_out = x_conv_out.view(-1)
        return F.sigmoid(self.linear(x_conv_out))


cnn_ = CNNClassifier(embed_size=768, kernel_size=5, num_filter=2)
for sents, tgt in batch_iter(list(zip(x_train, y_train)), 12):
    print(type(sents), type(sents[0]))
    # print(sents[0][0].shape)
    # sents = torch.tensor(sents)
    cnn_.forward(sents)
# (12, 103, 768)
# class LogisticModel(nn.Module):
#     def __init__(self, embed_size):
#         super(LogisticModel, self).__init__()
#         self.embed_size = embed_size
#         self.linear = nn.Linear(in_features=self.embed_size, out_features=1)

#     def forward(self, input):
#         return F.sigmoid(self.linear(input))



# model = LogisticModel(768)
# y = model.forward(x_train)
# print(y.shape)
