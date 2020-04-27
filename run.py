from utils import read_data, pad_sentences, batch_iter
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings
from cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from classifiers import CNNClassifier
from pipeline import Pipeline, TrainArgs

def run():

    device = 'cpu'

    embed = Embeddings(device)
    train_corpus = embed.convert_corpus(TRAIN_FILE_PATH, 100)
    test_corpus = embed.convert_corpus(TEST_FILE_PATH, 50)
    
    embed_dim = 768
    kernel_size = 5
    num_filters = 10
    cnn_clf = CNNClassifier(embed_dim, kernel_size, num_filters)
    optimizer_cls = optim.Adam
    loss_cls = nn.NLLLoss

    train_args = TrainArgs(epochs=10, device=device)
    pipeline = Pipeline(train_corpus, cnn_clf, loss_cls)
    pipeline.train_model(optimizer_cls, train_args=train_args)
    pipeline.evaluate(train_corpus)
    pipeline.evaluate(test_corpus)


if __name__ == '__main__':

    run()