from embedding import Embeddings
from transformers import *
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

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
print(max_length_train)
# x_train = torch.tensor(x_train)
# x_test = torch.tensor(x_test)

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
