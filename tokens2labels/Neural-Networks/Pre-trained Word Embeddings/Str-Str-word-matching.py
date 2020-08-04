### Import Packages
from typing import Iterator, List, Dict

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from allennlp.data import Instance
from allennlp.data.fields import TextField, ArrayField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import BooleanAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

### Read Data
class PosDatasetReader(DatasetReader):

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens_text: List[Token], tokens_label: List[Token], match: List[int] = None) -> Instance:
        text_field = TextField(tokens_text, self.token_indexers)
        label_field = TextField(tokens_label, self.token_indexers)
        fields = {"text": text_field, "label": label_field}

        if match is not None:
            match_field = ArrayField(np.array([int(match)]))
            fields["match"] = match_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:
        data = pd.read_pickle(file_path)
        for row_index, row in data.iterrows():
            text = row['text'].lower().split()
            label = row['label'].lower().split()
            match = row['match']
            yield self.text_to_instance([Token(word) for word in text], [Token(word) for word in label], match)

### NN model structure
class Net(Model):

    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2SeqEncoder, vocab: Vocabulary, padding: int) -> None:
        super().__init__(vocab)

        self.padding = padding

        self.word_embeddings = word_embeddings
        self.encoder = encoder

        self.linear_u = torch.nn.Linear(in_features = encoder.get_output_dim(), out_features = 1)
        self.linear_db = torch.nn.Linear(in_features = encoder.get_output_dim(), out_features = 1)

        self.bilinear = torch.nn.Bilinear(in1_features = self.padding, in2_features = self.padding, out_features = 1)

        self.sigmoid = torch.nn.Sigmoid()

        self.accuracy = BooleanAccuracy()

        self.loss = torch.nn.BCELoss()

    def forward(self, text: Dict[str, torch.Tensor], label: Dict[str, torch.Tensor], match: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        batch_size = text['tokens'].size()[0]

        text_pad = torch.zeros(batch_size, self.padding).long()#.cuda()
        label_pad = torch.zeros(batch_size, self.padding).long()#.cuda()

        text_pad[:,:text['tokens'].size()[1]] = text['tokens']
        label_pad[:,:label['tokens'].size()[1]] = label['tokens']

        text['tokens'] = text_pad
        label['tokens'] = label_pad

        text_mask = get_text_field_mask(text)
        label_mask = get_text_field_mask(label)

        text_embeddings = (self.word_embeddings(text))
        label_embeddings = (self.word_embeddings(label))

        text_encoder_out = F.relu(self.encoder(text_embeddings, text_mask))
        label_encoder_out = F.relu(self.encoder(label_embeddings, label_mask))

        text_linear = (self.linear_u(text_encoder_out))
        label_linear = (self.linear_db(label_encoder_out))

        text_flatten = text_linear.view([batch_size,-1])
        label_flatten = label_linear.view([batch_size,-1])

        bilinear = self.bilinear(text_flatten, label_flatten)

        match_prob = self.sigmoid(bilinear)

        match_result = match_prob.clone()

        for result in match_result:
            if result[0] < 0.5:
                result[0] = 0
            else:
                result[0] = 1

        output = {'match_output':match_result}

        if match is not None:
            self.accuracy(match_result, match)
            output['loss'] = self.loss(match_prob, match)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy':self.accuracy.get_metric(reset)}


### Main Program
if __name__ == "__main__":

    torch.manual_seed(1)

    ### Read Data

    reader = PosDatasetReader()

    token = 'exercise'
    label = 'exercise'

    train_path = '../data/{}2{}-match-train-df.pkl'.format(token, label)
    test_path = '../data/{}2{}-match-test-df.pkl'.format(token, label)

    train_dataset = reader.read(train_path)
    validation_dataset = reader.read(test_path)

    vocab = Vocabulary.from_instances(train_dataset+validation_dataset)

    ### Build Model

    EMBEDDING_DIM = 64 #between 50 and 300
    HIDDEN_DIM = 128  #between 50 and 300

    EMBEDDING = '../../../../word2vec.bin' # TODO: try glove, word2vec, fastText

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),embedding_dim=EMBEDDING_DIM, pretrained_file=EMBEDDING) # use GloVe and Word2Vec

    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(torch.nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first = True, num_layers = 1)) #LSTM or GRU and num_layers = #

    max_length = 10 #60 for sentence and 10 for exercise

    model = Net(word_embeddings, lstm, vocab, max_length)

    ### Check Cuda
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    ### Train Model

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9) #optim.Adam(same), play with lr and momentum (SGD ony)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("text", "num_tokens"), ("label", "num_tokens")]) # 32 speed - 64 precission

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=100,
                      num_epochs=10000,
                      cuda_device=cuda_device) #Select patience and play with number of epochs

    results = trainer.train()
    print(results)
