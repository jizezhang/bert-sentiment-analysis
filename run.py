from utils import read_data, pad_sentences, batch_iter
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings

def run():

    train_sents, train_score = read_data(TRAIN_FILE_PATH, 12)

    embed = Embeddings()
    train_sent_tensors = pad_sentences([embed.sentence2matrix(sent) for sent in train_sents])

    for sents, scores in batch_iter(list(zip(train_sent_tensors, train_score)), 3):
        print(sents.shape, scores.shape)


if __name__ == '__main__':

    run()