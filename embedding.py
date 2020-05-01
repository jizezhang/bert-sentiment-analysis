# -*- coding: utf-8 -*-
from transformers import BertModel
import torch
import torch.nn as nn
from utils import read_data


class Embeddings(nn.Module):

    def __init__(self):
        super().__init__()
        self._bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self._bert_model.eval()

    def forward(self, x):
        tokens_ids, attn_mask = x
        return self.sentence2matrix(tokens_ids, attn_mask)

    def sentence2matrix(self, tokens, attn_masks):
        with torch.no_grad():
            encoded_layers = self._bert_model(tokens, attention_mask=attn_masks)
        return torch.sum(torch.stack(encoded_layers[-1][-4:]), 0)