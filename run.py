from utils import read_data, pad_sentences, batch_iter
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings
from cnn import CNN
from classifiers import CNNClassifier

def run():

    train_sents, train_score = read_data(TRAIN_FILE_PATH, 12)

    embed = Embeddings()
    train_sent_tensors = pad_sentences([embed.sentence2matrix(sent) for sent in train_sents])

    embed_dim = train_sent_tensors[0].shape[-1]
    kernel_size = 5
    num_filters = 10
    # cnn = CNN(embed_dim, kernel_size, num_filters)
    cnn_clf = CNNClassifier(embed_dim, kernel_size, num_filters)

    for sents, scores in batch_iter(list(zip(train_sent_tensors, train_score)), 3):
        output = cnn_clf.forward(sents)
        print(output.shape)


if __name__ == '__main__':

    run()