from constants import TEXT_COL, SCORE_COL
import pandas as pd
import torch
import math
import numpy as np

def read_data(path, num_data=None):
   df = pd.read_csv(path)
   text = df[TEXT_COL].values
   score = df[SCORE_COL].values

   if num_data is not None:
      return text[:num_data], score[:num_data]
   else:
      return text, score

def pad_sentences(sentences, word_embed_dim=None):
    """
    :param sentences: List[tensor], where tensor has shape (num_tokens, num_embed)
    """
    if word_embed_dim is None:
        word_embed_dim = sentences[0].shape[1]

    max_sent_len = max([sent.shape[0] for sent in sentences])
    zero_padding_vec = torch.tensor([0.0] * word_embed_dim)
    return [torch.cat((sent, zero_padding_vec.repeat(max_sent_len - sent.shape[0], 1))) for sent in sentences]

def batch_iter(data, batch_size=None, shuffle=False, pad_batch=True):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    """
    if batch_size is None:
        num_batch = 1
        batch_size = len(data)
    else:
        num_batch = math.ceil(len(data) / batch_size)
    
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)
    
    for i in range(num_batch):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        samples = [data[idx] for idx in indices]
        inputs = [e[0] for e in samples]
        if pad_batch:
            inputs = pad_sentences(inputs)
        targets = [e[1] for e in samples]
        yield torch.stack(inputs), torch.tensor(targets)
