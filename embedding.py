# -*- coding: utf-8 -*-
from transformers import BertModel, BertTokenizer
import torch
from utils import read_data
from tqdm import tqdm


class Embeddings:

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self._bert_model.eval()

    def tokenize(self, sentence):
        """
        :param sentence: input sentence ['str']
        :return: tokenized sentence based on word piece model ['List']
        """
        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self._tokenizer.tokenize(marked_sentence)
        return tokenized_text

    def get_bert_embeddings(self, sentence):
        """
        :param sentence: input sentence ['str']
        :return: BERT pre-trained hidden states (list of torch tensors) ['List']
        """
        # Predict hidden states features for each layer

        tokenized_text = self.tokenize(sentence)
        indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            encoded_layers = self._bert_model(tokens_tensor, token_type_ids=segments_tensors)

        return encoded_layers[-1][0:12]

    def sentence2matrix(self, sentence, use_last_4=True):
        """
        :param sentence: input sentence ['str']
        :return: sentence [List]
        """
        encoded_layers = self.get_bert_embeddings(sentence)
        
        if not use_last_4:
            # using the last layer embeddings
            return encoded_layers[-1][0]

        else:
            return torch.sum(torch.cat(encoded_layers[-4:]), 0)

    def convert_corpus(self, path, num_entries):
        sents, scores = read_data(path, num_entries)
        sent_tensors = [self.sentence2matrix(sent) for sent in tqdm(sents)]
        corpus = list(zip(sent_tensors, scores))
        return corpus