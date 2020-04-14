import math
import numpy as np
import torch
def pad_sents(sents, embed_size=768, pad_token=0):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)

    for s in sents:
        padded = [[pad_token] * embed_size] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    
    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        # examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt = [e[1] for e in examples]
        src_sents = torch.cat(src_sents)
        # tgt = torch.cat(tgt)
        yield src_sents, tgt

