class Embeddings:
    LAST_LAYER = 1
    LAST_4_LAYERS = 2
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

    def sentence2vec(self, sentence, layers):
        """

        :param sentence: input sentence ['str']
        :param layers: parameter to decide how word embeddings are obtained ['str]
            1. 'last' : last hidden state used to obtain word embeddings for sentence tokens
            2. 'last_4' : last 4 hidden states used to obtain word embeddings for sentence tokens

        :return: sentence vector [List]
        """
        encoded_layers = self.get_bert_embeddings(sentence)
        
        if layers == 1:
            # using the last layer embeddings
            token_embeddings = encoded_layers[-1]
            # summing the last layer vectors for each token
            sentence_embedding = torch.mean(token_embeddings, 1)
            return sentence_embedding.view(-1).tolist()

        elif layers == 2:
            token_embeddings = []
            tokenized_text = self.tokenize(sentence)

            batch_i = 0
            # For each token in the sentence...
            for token_i in range(len(tokenized_text)):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(list(vec.numpy()))

                token_embeddings.append(hidden_layers)

            # using the last 4 layer embeddings
            token_vecs_sum = []

            # For each token in the sentence...
            for token in token_embeddings:
                # Sum the vectors from the last four layers.
                sum_vec = np.sum(token[-4:], axis=0)

                # Use `sum_vec` to represent `token`.
                token_vecs_sum.append(list(sum_vec))

            # summing the last layer vectors for each token
            sentence_embedding = np.mean(token_vecs_sum, axis=0)
            return sentence_embedding.ravel().tolist()
