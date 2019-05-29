import json
import time

from flair.data import Sentence
from flair.embeddings import BertEmbeddings

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from utils import *

# init multilingual BERT
bert_embedding = BertEmbeddings('bert-base-multilingual-cased')

# Takes a Tokenized sentence computed by the flair.Sentence() function
# as input and outputs a torch.Tensor of shape #num_tokens x bert_embedding_size
def embedding(sentence):
    bert_embedding.embed(sentence)
    return torch.Tensor([list(word.embedding) for word in sentence])

# Defines an LSTM module and output one unique prediction with its last linear layer
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers , label_size, 
                 batch_size, use_gpu):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.layers = layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layers)
        self.out = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.layers, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.layers, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layers, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.layers, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, sentence):
        embeds = sentence
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.out(lstm_out[-1])
        return y

# Defines a regular fully connected neural net with 1 input layer 1 output layer
# as well as 3 hidden layers. The output is going to be a unique number
class HybridDense(nn.Module):
    def __init__(self, input_dim, hidden_dim, label_size):
        super(HybridDense, self).__init__()
        self.input_dim = input_dim
        self.label_size = label_size
        
        self.input = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, label_size)
        
    def forward(self, meta_data):
        x = self.input(meta_data)
        x = self.hidden(x)
        x = self.hidden(x)
        x = self.hidden(x)
        y = self.out(x)
        return y

# This model combines the two previous modules LSTM on one part and Dense on the other
# It will process in the LSTM the sentence processing and prediction part
# The Dense network will process the metadata prediction part on its side
# We eventually process both outputs in a final fully connected layer to predict our final answer
class HybridModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, layers , label_size, 
                 batch_size, use_gpu, input_dim, hidden_dim2): # LSTM => DENSE
        super(HybridModel,self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.label_size = label_size
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        
        self.input_dim = input_dim
        self.hidden_dim2 = hidden_dim2
        
        self.lstm_model = LSTMClassifier(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            label_size=self.label_size,
            batch_size=self.batch_size,
            use_gpu=self.use_gpu
            ).to(device)
        
        self.dense = HybridDense(
            input_dim = self.input_dim, 
            hidden_dim = self.hidden_dim2,
            label_size = self.label_size
            ).to(device)
        
        self.final_layer = nn.Linear(2, label_size)
        
    def forward(self, sentence, meta_data):
        lstm_out = self.lstm_model(sentence).squeeze_(0)
        dense_out = self.dense(meta_data)
        out = torch.cat((lstm_out,dense_out), 0)
        y = self.final_layer(out)
        return y

# We first count the probability of getting a 1 as a target
count = 0
total = len(list(data.keys()))
for key, val in data.items():
    try:
        if int(val['is_best_answer']) == 1:
            count += 1
    except:
        continue
PART = count/total

# We then compute the threshold we need to approximatily have 1/2 of the time
# a target that is 1 so as to balance the training.
# The computation is biased since we drop sequences that are longer than 512 
# for BERT Embedding. Meaning we have to even lower the probability of training
# on target equal == 0 each time we encounter one.
prob = 1-(PART)

# This function takes as inputs: model, optimizer criterion and data
# It outpout the loss evolution as well as well as the number of different models 
# created by the checkpoints. It trains the model with and SGD method.
def stochastic_train(model, optimizer, criterion, clip, data_source):
    
    losses = {}
    iter_count = 0
    mod_count = 0
    
    model.train()
    
    for key in data_source.keys():
        iter_count +=1
        
        start_time = time.time()
        try:
            sentence, meta_data = make_input(data_source, key)
            trg = int(data_source[key]['is_best_answer'])
        except:
            continue
        
        # Got the minibatch
        optimizer.zero_grad()
        if (trg == 0) and (np.random.rand() < prob):
            continue
        if len(sentence) > 512:
            continue
        # Launching BERT embedding
        sentence = embedding(Sentence(sentence)).to(device)
        
        meta_data = torch.Tensor(meta_data).to(device)
        trg = torch.Tensor([trg]).to(device)
        
        # Launching model forward
        output = model(sentence, meta_data)
        
        model.zero_grad()
        
        
        loss = criterion(torch.sigmoid(output), trg)
        loss.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        end_time = time.time()
        batch_mins, batch_secs = batch_time(start_time, end_time)
        
        print('Time: {}:{}\nLoss: {}'.format(batch_mins,batch_secs,loss.item()))
        losses[key] = loss.item()
        
        
        if iter_count% 300 == 0:
            with open("drive/My Drive/MSc BD & BA/Period 3/NLPy/results.json", 'w') as f:
                json.dump(losses, f)
            torch.save(model.state_dict(), 
                       'drive/My Drive/MSc BD & BA/Period 3/NLPy/model{}.pt'.format(
                       mod_count))
            mod_count +=1
        
    return losses, mod_count   

# This function allows the user to control the overall programm.
# One can chose wether to execute this file locally or on google colab
# To train with or without a GPU. To predict or not a given sentence.
# To load or not a given model checkpoint.
def execution(
        env="drive",
        path="data.json",
        use_gpu=True,
        train=True,
        load=False,
        predict=False,
        model_path='',
        sentence='',
        model_id='',
        meta_data=[0]*8,
        **kwargs):

    if env=="drive":
        from google.colab import drive
        drive.mount('/content/drive')
        path = "drive/My Drive/MSc BD & BA/Period 3/NLPy/data.json"
    elif env=="local":
        path="data/teach_data/data.json"
    
    with open(path, 'r') as f:
            data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(
            embedding_dim=3072,
            hidden_dim=128,
            layers=3,
            label_size=1,
            batch_size=1,
            use_gpu=use_gpu,
            input_dim=8,
            hidden_dim2=7
            ).to(device)
    

    model.apply(init_weights)
    print("The model has {} trainable parameters".format(count_parameters(model
                                                                         )))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    clip = 1

    if train:
        train_loss, mod_count = stochastic_train(model, optimizer, criterion, 
                                                 clip, data)

    if load:
        model.load_state_dict(torch.load(model_path+'model{}.pt'.format(
            model_id)))

    if predict:
        with torch.no_grad():
            out = model(embedding(Sentence(sentence)),torch.Tensor(meta_data))
            out = torch.sigmoid(out)
        print(out)

if __name__=="__main__":
    # This file is optimized for Google Colab execution
    execution()
