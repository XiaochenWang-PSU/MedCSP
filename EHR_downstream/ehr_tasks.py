
from sklearn import preprocessing
import numpy as np
from model import LSTM_Encoder, MLP_Encoder, MLP_Decoder, Simple, Readm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sparse
from tqdm import tqdm
import pandas as pd
import pickle
import ast
from info_nce import InfoNCE
import time
import torch.distributed as dist
from datetime import datetime
import time
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score,precision_recall_curve, auc
import AUROC_models as models
import sys
from trainer import *
import csv

print(sys.argv[2:] )
EPOCHS,LR, BATCH, TASK = sys.argv[2:] 
EPOCHS,LR, BATCH, TASK = int(EPOCHS), float(LR), int(BATCH),  str(TASK)



device = torch.device("cuda:1")
data = pickle.load(open(TASK + "_pred.p", 'rb'))
print("Data size:", len(data))

def collate_batch_ehr(batch_data):
    # Assuming icd and drug are already tensors or compatible formats

    # Convert COO sparse matrix to dense tensor for X and S
    X = torch.stack([torch.tensor(k[0]) for k in batch_data]).float().to(device)
    S = torch.stack([torch.tensor(k[1]) for k in batch_data]).float().to(device)


    label = torch.tensor([i[2] for i in batch_data]).float().to(device)

    return [X, S,label]
def collate_batch_readm(batch_data):
    # Assuming icd and drug are already tensors or compatible formats

    # Convert COO sparse matrix to dense tensor for X and S
    
    icd = torch.stack([torch.tensor(i[0]) for i in batch_data]).float().to(device)
    drug = torch.stack([torch.tensor(i[1]) for i in batch_data]).float().to(device)
    X = torch.stack([torch.stack([torch.tensor(j.todense()) for j in k[2]], dim=0) for k in batch_data]).float().to(device)
    S = torch.stack([torch.tensor(k[3][0].todense()) for k in batch_data]).float().to(device)

    text = torch.stack([i[4] for i in batch_data]).squeeze(1).float().to(device)
    label = torch.tensor([i[5] for i in batch_data]).float().to(device)



    return [X, S,icd, drug, text, label]

split_mark = int(len(data)*0.8), int(len(data)*0.9)

criterion = torch.nn.BCEWithLogitsLoss()
if TASK != "readm":
    model = torch.load("pretrained_icu.p",map_location="cpu").to(device).float()    
    test = DataLoader(data[split_mark[1]:], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_ehr)
    train = DataLoader(data[:split_mark[0]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_ehr)
    valid = DataLoader(data[split_mark[0]:split_mark[1]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_ehr)
    
    model, _ = icu_train(model, train, valid, test,EPOCHS, LR, BATCH, device, TASK, patience=3)
    
    
    
    
    print(list(eval_metric(test, model,  device))[:-1])
    

else:
    model = torch.load("pretrained_readm.p",map_location="cpu").to(device).float()    
    test = DataLoader(data[split_mark[1]:], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_readm)
    train = DataLoader(data[:split_mark[0]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_readm)
    valid = DataLoader(data[split_mark[0]:split_mark[1]], batch_size = BATCH, shuffle = True, collate_fn=collate_batch_readm)
    
    model, _ = readm_train(model, train, valid, test,EPOCHS, LR, BATCH, device, TASK, patience=3)
    
    
    
    
    print(list(eval_metric_readm(test, model,  device))[:-1])
    

globals().clear()