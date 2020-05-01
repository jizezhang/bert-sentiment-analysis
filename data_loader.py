from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import read_data
import torch

class BertDataset(Dataset):

    def __init__(self, file_path, num_entries):
        super().__init__()
        self.sents, self.scores = read_data(file_path, num_entries)
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_entries = len(self.sents)
        self.pad_token_id = self._tokenizer.convert_tokens_to_ids('[PAD]')

    def tokenize(self, sentence, score):
        """
        :param sentence: input sentence ['str']
        :return: tokenized sentence based on word piece model ['List']
        """
        marked_sentence = "[CLS] " + sentence + " [SEP]"
        tokenized_text = self._tokenizer.tokenize(marked_sentence)
        indexed_tokens = self._tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_ids_tensor = torch.tensor(indexed_tokens)

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = torch.ones([len(tokenized_text)])
        return tokens_ids_tensor, attn_mask, score

    def __getitem__(self, index):
        return self.tokenize(self.sents[index], self.scores[index])

    def __len__(self):
        return self.num_entries


class SentDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.pad_collate)
        self.pad_token_id = self.dataset.pad_token_id

    def pad_collate(self, batch):
        (sent, masks, y) = zip(*batch)
        sent_pad = pad_sequence(sent, batch_first=True, padding_value=self.pad_token_id)
        masks_pad = pad_sequence(masks, batch_first=True, padding_value=0)
        return sent_pad, masks_pad, torch.tensor(y)


