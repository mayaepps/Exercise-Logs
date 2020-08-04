'''
Description: Tagging of exercise logs and feelings with PyTorch neural nets.
Author: Mandy korpusik
Date: 4/10/2020

References:
https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling/blob/master/train.py
'''

import conlleval
import pandas as pd
import numpy as np
import mmap
import io
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn_crfsuite import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


class SentenceGetter(object):
    '''Loads all sentences from the Pandas DataFrame as (word, POS, tag).'''
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
        self.grouped = self.data.groupby('Sentence #').apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


class FFTagger(nn.Module):
    '''
    Embedding followed by linear hidden layer and linear output layer with softmax.
    Parameters: embedding_dim, hidden_dim
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(FFTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if EMBEDDING:
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding.weight.requires_grad = False

        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        if NUM_LAYERS > 1:
            self.hidden2 = nn.Linear(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space.
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        hidden_out = self.hidden(embeds.view(len(sentence), 1, -1))
        if NUM_LAYERS > 1:
            hidden_out = self.hidden2(hidden_out.view(len(sentence), -1))
        tag_space = self.hidden2tag(hidden_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTMTagger(nn.Module):
    '''
    Single layer LSTM with embedding layer and linear output layer with softmax.
    Parameters: embedding_dim, hidden_dim
    '''
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class CNNTagger(nn.Module):
    '''
    Single layer CNN with given number of filters and filter sizes (i.e., width).
    Parameters: embedding_dim, num_filters, filter_sizes, drp
    '''
    def __init__(self, embedding_dim, num_filters, vocab_size, tagset_size, filter_sizes=[1,2,3,5], drp=0.1):
        super(CNNTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs1 = nn.ModuleList([nn.Conv1d(1, num_filters, kernel_size) for kernel_size in filter_sizes])
        self.dropout = nn.Dropout(drp)
        self.hidden2tag = nn.Linear(num_filters * len(filter_sizes), tagset_size)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        cnn_out = [F.relu(conv(embeds.view(len(sentence), 1, -1))) for conv in self.convs1]
        max_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in cnn_out]
        combined_out = torch.cat(max_out, 1)
        x = self.dropout(combined_out)
        tag_space = self.hidden2tag(x)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def prepare_sequence(sentence, word_to_ix, tag_to_ix):
    '''Converts a sequence of (word, POS, tag) tuples to word and tag indices.'''
    word_idxs = []
    tag_idxs = []
    for word, POS, tag in sentence:
        if LOWER:
            word = word.lower()
        word_idxs.append(word_to_ix[word])
        tag_idxs.append(tag_to_ix[tag])
    return torch.tensor(word_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)


def load_tag_dict(classes):
    '''Returns a dictionary mapping each tag to its index.'''
    tag_to_ix = {}
    ix_to_tag = {}
    for tag in classes:
        tag_to_ix[tag] = len(tag_to_ix)
        ix_to_tag[tag_to_ix[tag]] = tag
    return tag_to_ix, ix_to_tag


def load_word_dict(sentences):
    '''Returns a dictionary mapping each word to its index in the vocabulary.'''
    word_to_ix = {}
    for sent in sentences:
        for word, POS, tag in sent:
            if LOWER:
                word = word.lower()
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def load_word2vec(fname='data/word2vec.bin'):
    '''Loads word2vec word vectors.'''
    fin = open(fname, 'rb')
    header = fin.readline()
    vocab_size, vector_size = map(int, header.split())

    fbuf = mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
    vecs_dict = {}
    binary_len = np.dtype(np.float32).itemsize * vector_size
    for i in range(vocab_size):
        word = b''.join(iter(partial(fin.read, 1), b' ')).strip().decode('utf-8')
        if LOWER:
            word = word.lower()
        vec = np.frombuffer(fbuf, dtype=np.float32, offset=fin.tell(), count=vector_size)
        vecs_dict[word] = vec
        fin.seek(binary_len, io.SEEK_CUR)
    return vecs_dict


def load_fastText(fname='data/fastText.vec'):
    '''Loads FastText word vectors.'''
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        if LOWER:
            line = line.lower()
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data


def load_glove(fname='data/glove.6B.200d.txt'):
    '''Loads Glove word vectors.'''
    vecs = {} # Maps words to vecs.
    for line in open(fname).readlines():
        line = line.strip().split()
        word = line[0]
        if LOWER:
            word = word.lower()
        vec = [float(val) for val in line[1:]]
        vecs[word] = vec
    return vecs


if __name__ == '__main__':

    ### LOADING THE DATA ###
    df = pd.read_csv('trainDataWithPOS.csv', encoding = "ISO-8859-1")
    dfTest = pd.read_csv('testDataWithPOS.csv', encoding = "ISO-8859-1")
    classes = np.unique(df.Tag.values).tolist()
    tag_to_ix, ix_to_tag = load_tag_dict(classes)
    NUM_LAYERS = 1 # TODO: try different number of layers (e.g., 2)
    OUTPUT_DIM = len(classes)
    EMBEDDING_DIM = 50 # TODO: try different embedding dimensions (e.g., 100, 200, 300, etc.)
    HIDDEN_DIM = 64 # TODO: try different hidden dimensions (e.g., 32, 128, 256)
    LOWER = True
    EMBEDDING = None # TODO: try glove, word2vec, fastText
    if EMBEDDING == "glove":
        vecs = load_glove()
        EMBEDDING_DIM = 200
    elif EMBEDDING == "word2vec":
        vecs = load_word2vec()
        EMBEDDING_DIM = 300
    elif EMBEDDING == "fastText":
        vecs = load_fastText()
        EMBEDDING_DIM = 300
    
    sentences = SentenceGetter(df).sentences
    testSentences = SentenceGetter(dfTest).sentences
    word_to_ix = load_word_dict(sentences)

    train_sents = sentences
    test_sents = testSentences
    train_sents, val_sents = train_test_split(train_sents, test_size=0.2, random_state=0, shuffle=True)

    ### TRAINING THE MODEL ###
    if EMBEDDING:
        embedding_matrix = np.zeros((len(word_to_ix), EMBEDDING_DIM))
        for word in word_to_ix:
                token_index = word_to_ix[word]
                if word in vecs:
                    embedding_matrix[token_index] = vecs[word]
                     
    model = FFTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    # Determine whether using CCPU or GPU for training.
    device = 'cpu'
    if torch.cuda.is_available():
        model.cuda()
        device = 'cuda'

    loss_function = nn.NLLLoss()
    # TODO: try different learning rates (e.g., lr=0.001, 0.001, etc.)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    prev_loss = None
    epochs = 1000

    for epoch in range(epochs):
        for sentence in train_sents:
            # Step 1. Clear out any accumulated gradients.
            model.zero_grad()

            # Step 2. Convert data into tensors of word and tag indices.
            x, y = prepare_sequence(sentence, word_to_ix, tag_to_ix)

            # Step 3. Run the forward pass
            tag_scores = model(x.to(device))

            # Step 4. Compute the loss and gradients, and update the parameters.
            loss = loss_function(tag_scores, y.to(device))
            loss.backward()
            optimizer.step()

        val_losses = []
        with torch.no_grad():
            for sentence in val_sents:
                x, y = prepare_sequence(sentence, word_to_ix, tag_to_ix)
                tag_scores = model(x.to(device))
                loss = loss_function(tag_scores, y.to(device))
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
        print('epoch {}, loss {}'.format(epoch, val_loss))

        # Early stopping: break out if loss after 10 epochs is worse.
        if prev_loss and val_loss > prev_loss:
            break

        prev_loss = val_loss


    ### EVALUATION ###
    with torch.no_grad():
        true_y = []
        pred_y = []
        # Get argmax (i.e., tag with highest prob) for each word in test set.
        for sentence in val_sents:
            x, y = prepare_sequence(sentence, word_to_ix, tag_to_ix)
            tag_scores = model(x.to(device)).cpu().data.numpy()
            true_y.extend(y.data.numpy())
            pred_y.extend(np.argmax(tag_scores, axis=1))

        # Compute precision, recall, and F1 score (with and without 'O' tag).
        prec, rec, f1, _ = precision_recall_fscore_support(true_y, pred_y)
        for label, prec, rec, f1 in zip(classes, prec, rec, f1):
            print(label, prec, rec, f1)
        print('weighted average:', precision_recall_fscore_support(true_y, pred_y, average='weighted'), '\n')

        classes.remove('O')
        labels = [i for i in range(len(classes))]
        prec, rec, f1, _ = precision_recall_fscore_support(true_y, pred_y, labels=labels)
        for label, prec, rec, f1 in zip(classes, prec, rec, f1):
            print(label, prec, rec, f1)
        print(precision_recall_fscore_support(true_y, pred_y, labels=labels, average='weighted'))

        # Use CoNLL-style evaluation (per-entity F1, rather than per-token).
        true_y = [ix_to_tag[ix] for ix in true_y]
        pred_y = [ix_to_tag[ix] for ix in pred_y]
        conlleval.evaluate(true_y, pred_y)


