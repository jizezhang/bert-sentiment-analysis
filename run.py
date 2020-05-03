from utils import read_data
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings
from cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from classifiers import CNNClassifier
from pipeline import Pipeline, TrainArgs
from data_loader import BertDataset, SentDataLoader

def run():

    device = 'cpu'

    train_data = BertDataset(TRAIN_FILE_PATH, num_entries=500)
    test_data = BertDataset(TEST_FILE_PATH, num_entries=100)
    batch_size = 5
    train_dataloader = SentDataLoader(train_data, batch_size=batch_size)
    test_dataloader = SentDataLoader(test_data, batch_size=batch_size)


    embedding = Embeddings()
    
    embed_dim = 768
    kernel_size = 5
    num_filters = 10
    cnn_clf = CNNClassifier(embedding, embed_dim, kernel_size, num_filters)
    optimizer_cls = optim.Adam
    loss_cls = nn.NLLLoss

    train_args = TrainArgs(epochs=5, device=device)
    pipeline = Pipeline(train_dataloader, cnn_clf, loss_cls)
    pipeline.train_model(optimizer_cls, train_args=train_args)

    train_dataloader = SentDataLoader(train_data, batch_size=batch_size)
    # print('evaluate train:')
    # pipeline.evaluate(train_dataloader)
    print('evaluate test:')
    pipeline.evaluate(test_dataloader)


if __name__ == '__main__':

    run()