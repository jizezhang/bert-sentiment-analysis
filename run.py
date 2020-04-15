from utils import read_data, pad_sentences
from constants import TRAIN_FILE_PATH, TEST_FILE_PATH
from embedding import Embeddings

def run():

    train_sents, train_score = read_data(TRAIN_FILE_PATH, 2)

    embed = Embeddings()
    train_sent_tensors = pad_sentences([embed.sentence2matrix(sent) for sent in train_sents])
    print(len(train_sent_tensors), train_sent_tensors[1].shape)


if __name__ == '__main__':

    run()