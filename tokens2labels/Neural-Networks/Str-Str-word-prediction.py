### Import Packages
from typing import Iterator, List, Dict

from math import floor  

import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader

from allennlp.common.file_utils import cached_path

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from allennlp.data.vocabulary import Vocabulary

from allennlp.models import Model

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask

from allennlp.training.metrics import CategoricalAccuracy

from allennlp.data.iterators import BucketIterator

from allennlp.training.trainer import Trainer

### Read Data
class PosDatasetReader(DatasetReader):
    
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, tokens_text: List[Token], label: str = None) -> Instance:
        text_field = TextField(tokens_text, self.token_indexers)
        fields = {'text': text_field}

        if label is not None:
            label_field = LabelField(label)
            fields['label'] = label_field
            

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        data = pd.read_pickle(file_path)
        for row_index, row in data.iterrows():
            text = row['text'].lower().split()
            label = row['label']
            yield self.text_to_instance([Token(word) for word in text], label)

### NN model structure
class Net(Model):
    
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab: Vocabulary, n_classes: int) -> None:
        super().__init__(vocab)
                
        self.word_embeddings = word_embeddings

        self.encoder = encoder
        
        self.linear = torch.nn.Linear(in_features = encoder.get_output_dim(), out_features = n_classes)

        self.accuracy = CategoricalAccuracy()
        
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, text: Dict[str, torch.Tensor], label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        mask = get_text_field_mask(text)
        #print('Text mask size: {}'.format(mask.size()))

        embeddings = self.word_embeddings(text)
        #print('Text embeddings size: {}'.format(embeddings.size()))

        encoder_out = self.encoder(embeddings, mask)
        #print('Text Encoder size: {}'.format(encoder_out.size()))

        logits = self.linear(encoder_out)
        #print('Text logits size: {}'.format(logits.size()))

        output = {'logits':logits}

        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = self.loss(logits, label)
        #exit()
        return output
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy':self.accuracy.get_metric(reset)}


### Main Program
if __name__ == '__main__':
    
    torch.manual_seed(1)
    
    ### Read Data

    reader = PosDatasetReader()

    token = 'exercise'
    label = 'exercise'

    train_path = '../data/{}2{}-train-df.pkl'.format(token, label)
    test_path = '../data/{}2{}-test-df.pkl'.format(token, label)

    train_dataset = reader.read(train_path)
    validation_dataset = reader.read(test_path)

    vocab = Vocabulary.from_instances(train_dataset+validation_dataset)
    n_classes = vocab.get_vocab_size('labels')
    print(n_classes)
    exit()

    ### Build Model

    EMBEDDING_DIM = 64 #between 50 and 300
    HIDDEN_DIM = 128  #between 50 and 300

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=EMBEDDING_DIM) # use GloVe and Word2Vec 

    word_embeddings = BasicTextFieldEmbedder({'tokens': token_embedding})

    lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first = True, num_layers = 1)) #LSTM or GRU and num_layers = #

    model = Net(word_embeddings, lstm, vocab, n_classes)

    ### Check Cuda
    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    ### Train Model

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9) #optim.Adam(same), play with lr and momentum (SGD ony)

    iterator = BucketIterator(batch_size=32, sorting_keys=[('text', 'num_tokens')]) # 32 speed - 64 precission

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