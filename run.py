from utils import read_data, pad_sentences, batch_iter
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings
from cnn import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from classifiers import CNNClassifier
from pipeline import Pipeline

def run():

    train_sents, train_scores = read_data(TRAIN_FILE_PATH, 12)
    embed = Embeddings()
    train_sent_tensors = [embed.sentence2matrix(sent) for sent in train_sents]
    train_corpus = list(zip(train_sent_tensors, train_scores))
    # train_sent_tensors = pad_sentences(train_sent_tensors)

    embed_dim = train_sent_tensors[0].shape[-1]
    kernel_size = 5
    num_filters = 10
    cnn_clf = CNNClassifier(embed_dim, kernel_size, num_filters)
    optimizer_cls = optim.Adam
    loss_cls = nn.NLLLoss

    pipeline = Pipeline(train_corpus, cnn_clf, loss_cls)
    pipeline.train_model(optimizer_cls)

    # trained_model = train_model(train_corpus, cnn_clf, optimizer_cls, loss_cls)

    # test_sents, test_scores = read_data(TEST_FILE_PATH, 10)
    # test_sent_tensors = [embed.sentence2matrix(sent) for sent in test_sents]
    # _, predicted = torch.max(cnn_clf(test_sent_tensors), 1)






if __name__ == '__main__':

    run()