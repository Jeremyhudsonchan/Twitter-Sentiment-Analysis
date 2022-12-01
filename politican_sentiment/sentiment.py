import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda")

# import BERT-base pretrained model
bert = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)

# Load the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# sample data
text = ["Hi! This is a testing sample to see if this works. I hope it does!"]

# encode text
sent_id = tokenizer.batch_encode_plus(text, padding=True, return_token_type_ids=False)

# define model class
class BERT_Arch(nn.Module):
    
        def __init__(self, bert):
            super(BERT_Arch, self).__init__()
    
            self.bert = bert
    
            # dropout layer
            self.dropout = nn.Dropout(0.1)
    
            # relu activation function
            self.relu =  nn.ReLU()
    
            # dense layer 1
            self.fc1 = nn.Linear(768,512)
    
            # dense layer 2 (Output layer)
            self.fc2 = nn.Linear(512,2)
    
            #softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)
    
        #define the forward pass
        def forward(self, sent_id, mask):
    
            #pass the inputs to the model
            _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
    
            x = self.fc1(cls_hs)
    
            x = self.relu(x)
    
            x = self.dropout(x)
    
            # output layer
            x = self.fc2(x)
    
            # apply softmax activation
            x = self.softmax(x)
    
            return x

# pass the pre-trained BERT to our define architecture
model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

