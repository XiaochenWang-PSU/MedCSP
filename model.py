import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from open_clip import create_model_from_pretrained



from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# from llava.model.utils import *
import open_clip
import os, json




    

class LSTM_Encoder(nn.Module):
  def __init__(self, n_features, embedding_dim=32):
    super().__init__()
    self.n_features = n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=self.n_features,
      hidden_size=64,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=64,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

    self.norm = nn.LayerNorm(64)
    self.norm2 = nn.LayerNorm(512)
  def forward(self, x):
     

    dim_1 = x.shape[1]
    bz = x.shape[0]

    x = x.reshape(-1, x.shape[-2], x.shape[-1])

    x, (hidden_n, cell) = self.rnn1(x)
    x = torch.relu(x)

    x = self.norm(x)
    x, (hidden_n, cell) = self.rnn2(x)

    if dim_1 == 3:
        x = x.reshape(bz, 3, x.shape[-2], self.embedding_dim).mean(axis = 1)
        cell = cell.reshape(bz, 3, self.embedding_dim).mean(axis = 1)


    return self.norm2(x), cell



class MLP_Encoder(nn.Module):    
    def __init__(self, input_shape, emb_shape):
        super().__init__()

        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=64#input_shape
        )
        self.encoder_output_layer = nn.Linear(
            in_features=64, out_features=emb_shape
        )

        self.emb_shape = emb_shape
        self.norm = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(512)
    def forward(self, x):
        bz = x.shape[0]
        x = self.encoder_hidden_layer(x).squeeze().reshape(bz, 64)
        x = torch.relu(x)
        x = self.norm(x)
        x = self.encoder_output_layer(x).squeeze().reshape(bz, self.emb_shape)
        return self.norm2(x)
    
class MLP_Decoder(nn.Module):    
    def __init__(self, input_shape, emb_shape):
        super().__init__()

        # self.encoder_hidden_layer = nn.Sequential(nn.Linear(input_shape, int(input_shape/4)), nn.ReLU(), nn.Dropout(p = 0.5), nn.Linear(int(input_shape/4), emb_shape))
        self.encoder_hidden_layer = nn.Linear(input_shape, emb_shape)
        self.dropout = nn.Dropout(p = 0.2)
        self.emb_shape = emb_shape
    def forward(self, x):
        bz = x.shape[0]
        x = self.encoder_hidden_layer(x)
        x = x.squeeze().reshape(bz, self.emb_shape)

        return x
    

class MedCSP(nn.Module):    
    def __init__(self):
        super().__init__()

        


        self.clip, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')


        self.lstm_enc = LSTM_Encoder(1318, 512)
        self.icd_enc = MLP_Encoder(7686, 512)
        self.demo_enc = MLP_Encoder(73, 512)
        self.drug_enc = MLP_Encoder(1701, 512)
        
        
        
        self.norm1 = nn.LayerNorm(512)

        
        self.dropout1 = nn.Dropout(p = 0.2)


    def forward(self, x, modality):
        bz = x.shape[0]
        if modality == "time_series":
            x = torch.relu(self.lstm_enc(x)[1].squeeze())

        elif modality == "demo":
            x = torch.relu(self.demo_enc(x).squeeze())

        elif modality == "icd":
            x = torch.relu(self.icd_enc(x).squeeze())
            icd = self.dropout1(self.norm1(x))
        elif modality == "drug":
            x = torch.relu(self.drug_enc(x).squeeze())
        elif modality == "image":
            x = torch.relu(self.clip.encode_image(x).squeeze())
        elif modality == "text":
            x = torch.relu(self.clip.encode_text(x).squeeze())
        return self.dropout1(self.norm1(x))



class Downstream(nn.Module):    
    def __init__(self, model):
        super().__init__()

        self.mlp_enc = model.demo_enc
        self.mlp_dec = MLP_Decoder(512*2,1)

        self.lstm_enc = model.lstm_enc
        self.norm1 = nn.LayerNorm(512)

        self.dropout1 = nn.Dropout(p = 0.5)
    def forward(self, x, s):
        bz = x.shape[0]
        x = torch.relu(self.lstm_enc(x)[1].squeeze())
        s = torch.relu(self.mlp_enc(s).squeeze())
        output = self.dropout1(torch.cat((x,s), dim = -1))
        return self.mlp_dec(output).squeeze()
    
