#!/usr/bin/env python
# coding: utf-8

print('Loading packages...')

import time
import json
import torch
from math import exp
import pandas as pd
import torch.nn as nn
import torch.utils.data
from torch import optim
import torch.nn.functional as F
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Importing data...')

# Train and Test CSVs were created with the code in our "get_dataframe.ipynb"

train = pd.read_csv("../data/squad/train.csv")
test = pd.read_csv("../data/squad/test.csv")
train.drop('Unnamed: 0',inplace=True, axis=1)
test.drop('Unnamed: 0',inplace=True, axis=1)

# We'll only keep the context question and answer text for this task

X_train = train.drop(['id','answer_start', 'answer_stop','text', 'is_impossible'], axis=1)
y_train = train.text

X_test = test.drop(['id','answer_start', 'answer_stop','text', 'is_impossible'], axis=1)
y_test = test.text

# Compute tokens and store them in a list resulting in a list of lists

TOKENS_NOT_COMPUTED = False

if TOKENS_NOT_COMPUTED:
    print('Computing tokens...')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    tok_context_train = [tokenizer.tokenize(con) for con in X_train.context]
    tok_context_test = [tokenizer.tokenize(con) for con in X_test.context]

    tok_qs_train = [tokenizer.tokenize(qs) for qs in X_train.question]
    tok_qs_test = [tokenizer.tokenize(qs) for qs in X_test.question]

    tok_answer_train = [tokenizer.tokenize(con) for con in y_train.astype(str)]
    tok_answer_test = [tokenizer.tokenize(con) for con in y_test.astype(str)]


# Load tokens saved as json in the data folder

LOAD_VOCAB = True

if LOAD_VOCAB:
    print('Loading tokens...')

    with open('../data/squad/train/train_context_tok.json') as f:
        tok_context_train = json.load(f)

    with open('../data/squad/train/train_qs_tok.json') as f:
        tok_qs_train = json.load(f)

    with open('../data/squad/train/train_answer_tok.json') as f:
        tok_answer_train = json.load(f)

    ## Get json dicts for test

    with open('../data/squad/test/test_context_tok.json') as f:
        tok_context_test = json.load(f)

    with open('../data/squad/test/test_qs_tok.json') as f:
        tok_qs_test = json.load(f)

    with open('../data/squad/test/test_answer_tok.json') as f:
        tok_answer_test = json.load(f)

# Create a vocabulary class we use to structure our project


# Any word that has less than 100 occurences won't be added to the vocabulary
THRES = 100

class Vocab:
    def __init__(self, str_lst):
        self.str_lst = str_lst
        self.vocab, self.len_vocab = self.get_vocab()
        self.reversed_vocab = self.reverse_dict()
        self.ids_col = self.str_to_int(False)
        self.words_col = self.int_to_str(False)

    # We create a temporary dict

    def get_vocab(self):
        vocab = [item for sublist in self.str_lst for item in sublist]
        vocab_dict = {}
        counter = 1
        for token in vocab:
            if token not in vocab_dict.keys():
                vocab_dict[token] = {'id':counter, 'occurences':1}
                counter += 1
            else:
                vocab_dict[token]['occurences'] += 1

        length = [1 if vocab_dict[tok]['occurences'] > THRES else 0 for tok in vocab_dict.keys()]

        final_dict = {}
        counter = 1
        for iter in range(len(length)):
            if length[iter]:
                this = list(vocab_dict.keys())[iter]
                final_dict[this] = counter
                counter +=1

        final_dict['UNK'] = 0
        return final_dict, sum(length)

    def str_to_int(self, other):
        ints = []
        if other is False:
            for sentence in self.str_lst:
                sent = []
                for word in sentence:
                    try:
                        inti = self.vocab[word]
                    except KeyError:
                        inti = 0
                    sent.append(inti)
                ints.append(sent)
            return pd.Series(ints)
        else:
            for sentence in other:
                sent = []
                for word in sentence:
                    try:
                        inti = self.vocab[word]
                    except KeyError:
                        inti = 0
                    sent.append(inti)
                ints.append(sent)
            return pd.Series(ints)

    def reverse_dict(self):
        return {v: k for k, v in self.vocab.items()}

    def int_to_str(self, other):
        words = []
        if other is False:
            for sentence in self.ids_col:
                words.append([self.reversed_vocab[ids] for ids in sentence])
        else:
            for sentence in other:
                words.append([self.reversed_vocab[ids] for ids in sentence])
            
        return pd.Series(words)

class Squad:
    def __init__(self, context, qs):
        self.context = Vocab(context)
        self.qs = self.context.str_to_int(qs)

print('Building vocabulary...')

X_train_vocab = Squad(tok_context_train, tok_qs_train)
X_test_vocab = Squad(tok_context_test, tok_qs_test)

X_train["ids_context"] = X_train_vocab.context.ids_col
X_train["ids_qs"] = X_train_vocab.qs
y_train = X_train_vocab.context.str_to_int(tok_answer_train)

X_test["ids_context"] = X_test_vocab.context.ids_col
X_test["ids_qs"] = X_test_vocab.qs
y_test = X_test_vocab.context.str_to_int(tok_answer_test)


train_vocab_size = X_train_vocab.context.len_vocab+1


print('Train Vocab size is: {}'.format(train_vocab_size))

SAVE_TOKENS = False

if SAVE_TOKENS:
    print('Saving tokens...')
    ## Save vocabs and tokens for train
    with open('../data/train/train_qs_tok.json', 'w') as fp:
        json.dump(tok_qs_train, fp)

    with open('../data/train/train_context_vocab.json', 'w') as fp:
        json.dump(X_train_vocab.context.vocab, fp)

    with open('../data/train/train_context_tok.json', 'w') as fp:
        json.dump(tok_context_train, fp)

    with open('../data/train/train_answer_tok.json', 'w') as fp:
        json.dump(tok_answer_train, fp)
    
    ## Save vocabs and tokens for test

    with open('../data/test/test_qs_tok.json', 'w') as fp:
        json.dump(tok_qs_test, fp)

    with open('../data/test/test_context_vocab.json', 'w') as fp:
        json.dump(X_test_vocab.context.vocab, fp)

    with open('../data/test/test_context_tok.json', 'w') as fp:
        json.dump(tok_context_test, fp)

    with open('../data/test/test_answer_tok.json', 'w') as fp:
        json.dump(tok_answer_test, fp)


## Concat context and question separated from 4 zeros and pad them for TRAIN
print('Building tensors...')

input_tensor = []
for val in range(len(X_train.ids_context)):
    element = F.pad(torch.Tensor(X_train.ids_context[val] +[0,0,0]+ X_train.ids_qs[val]),
              pad=(0, 871-len(X_train.ids_context[val] +[0,0,0]+ X_train.ids_qs[val])), 
              mode='constant', value=0)
    input_tensor.append(element.numpy())
input_tensor = torch.LongTensor(input_tensor).to(device)


## Pad the target tensor
target_tensor = []
for val in range(len(y_train)):
    element = F.pad(torch.Tensor(y_train[val]),
              pad=(0, 68-len(y_train[val])),
              mode='constant', value=0)
    target_tensor.append(element.numpy())
target_tensor = torch.LongTensor(target_tensor).to(device)


## Concat context and question separated from 4 zeros and pad them for TEST
test_input_tensor = []
for val in range(len(X_test.ids_context)):
    element = F.pad(torch.Tensor(X_test.ids_context[val] +[0,0,0]+ X_test.ids_qs[val]),
              pad=(0, 871-len(X_test.ids_context[val] +[0,0,0]+ X_test.ids_qs[val])),
              mode='constant', value=0)
    test_input_tensor.append(element.numpy())
test_input_tensor = torch.LongTensor(test_input_tensor).to(device)


## Pad the target tensor
test_target_tensor = []
for val in range(len(y_test)):
    element = F.pad(torch.Tensor(y_test[val]),
              pad=(0, 68-len(y_test[val])),
              mode='constant', value=0)
    test_target_tensor.append(element.numpy())
test_target_tensor = torch.LongTensor(test_target_tensor).to(device)


train_data = TensorDataset(input_tensor, target_tensor)
trainDL = DataLoader(train_data, batch_size=871)

test_data = TensorDataset(test_input_tensor, test_target_tensor)
testDL = DataLoader(test_data, batch_size=871)

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
         
        outputs, (hidden, cell) = self.rnn(embedded)
        
        
        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, input_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True)
        self.out = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(0)
        #embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(input.float(), (hidden, cell))
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim,    "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,  "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg):
        batch_size = trg.shape[0]
        max_len = trg.shape[1]
        trg_vocab_size = train_vocab_size
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        
        hidden, cell = self.encoder(src)
        
        for t in range(1, max_len):
            
            output, hidden, cell = self.decoder(src, hidden, cell)
            outputs[t] = output
        return outputs


INPUT_DIM = train_vocab_size #len(INPUT.vocab)
OUTPUT_DIM = train_vocab_size #len(TARGET.vocab)
ENC_EMB_DIM = 64
DEC_EMB_DIM = input_tensor.size(-1)
HID_DIM = 64
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("The model has {} trainable parameters".format(count_parameters(model)))

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch[0]
        trg = batch[1]
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        output = output.view(-1, output.shape[-1])
        trg = trg.view(-1)

        loss = criterion(output, trg)
        print(loss.item())
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])
            trg = trg.view(-1)

            loss = criterion(output, trg)
            print(loss.item())
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 1
CLIP = 1

best_valid_loss = float('inf')

print('Launching training...')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, trainDL, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, testDL, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), '../models/seq2seq_model.pt')
    
    print('Epoch: {} | Time: {}m {}s'.format(epoch+1, epoch_mins, epoch_secs))
    print('\nTrain Loss: {} | Train PPL: {}'.format(train_loss, exp(train_loss)))
    print('\n Val. Loss: {} |  Val. PPL: {}'.format(valid_loss, exp(valid_loss)))
    
    
model.load_state_dict(torch.load('../models/seq2seq_model.pt'))
