
import json 
import pickle
import pandas as pd
import numpy as np
import sparse
import torch
import model
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score,precision_recall_curve, auc
import matplotlib.pyplot as plt
import math

import os



def eval_metric(eval_set, model,  device, encoder = 'normal'):
    
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, batch_data in enumerate(eval_set):
            X = batch_data[0].to(device)                                                                      
                                                              
            S = batch_data[1].to(device)
            labels = batch_data[-1].to(device)
            outputs = model(X,S).squeeze().to(device)                

            score = outputs
            score = score.data.cpu().numpy()
            labels = labels.data.cpu().numpy()

            if labels.shape[0] != 1:
                y_true = np.concatenate((y_true, labels))
                y_score = np.concatenate((y_score, score))
            else:
                y_true = np.array(list(y_true) + list(labels))
                y_score = np.array(list(y_score) + list([score]))

        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        loss = criterion(torch.from_numpy(y_score), torch.from_numpy(y_true))

    return  roc_auc, pr_auc, loss


def eval_metric_readm(eval_set, model,  device, encoder = 'normal'):
    
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        y_true = np.array([])
        y_pred = np.array([])
        y_score = np.array([])
        for i, batch_data in enumerate(eval_set):
            X = batch_data[0].to(device)                                                                      
                                                              
            S = batch_data[1].to(device)
            icd = batch_data[2].to(device)
            drug = batch_data[3].to(device)
            text = batch_data[4].to(device)
            labels = batch_data[-1].to(device)
            outputs = model(X,S, icd, drug, text).squeeze().to(device)                

            score = outputs
            score = score.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
 

            if labels.shape[0] != 1:
                
                y_true = np.concatenate((y_true, labels))
                y_score = np.concatenate((y_score, score))
            else:
                y_true = np.array(list(y_true) + list(labels))
                y_score = np.array(list(y_score) + list([score]))
        roc_auc = roc_auc_score(y_true, y_score)
        lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(lr_recall, lr_precision)
        loss = criterion(torch.from_numpy(y_score), torch.from_numpy(y_true))

    return  roc_auc, pr_auc, loss

def readm_train(model, train, valid, test, epoch, learn_rate, batch_size, device,  task, encoder = 'normal', patience = 10):
    

    model.train()
    aupr_list = []

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = learn_rate,
                                  weight_decay = 1e-3)
    
    
    

    roc_auc, pr_auc, valid_loss = eval_metric_readm(valid, model, device, encoder)
    best_dev = math.inf
    best_epoc = math.inf
    model.train()
    steps = 0

    for epoch in tqdm(range(epoch)):
        
        loss = 0
        
        for batch_idx, batch_data in enumerate(train):
            
                X = batch_data[0].to(device)
                S = batch_data[1].to(device)
                icd = batch_data[2].to(device)
                drug = batch_data[3].to(device)
                text = batch_data[4].to(device)

                label = batch_data[-1]
                optimizer.zero_grad()

                outputs = model(X,S, icd, drug, text).squeeze().to(device)

                train_loss = criterion(outputs, label)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
 
        print("Training Loss = ", loss)
        
        
    
        model.eval()
        

        roc_auc, pr_auc, valid_loss = eval_metric_readm(valid, model, device, encoder)
        print("Dev = ", valid_loss)
        if best_dev > valid_loss:
            best_dev = valid_loss
            best_epoc = epoch
            torch.save(model, "saved_model/" + str(task) + ".p")
        if epoch - best_epoc == patience:
            model = torch.load("saved_model/" + str(task) + ".p")
            break
        
        model.train()
    model = torch.load("saved_model/" + str(task) + ".p")
    return model, aupr_list



def icu_train(model, train, valid, test, epoch, learn_rate, batch_size, device,  task, encoder = 'normal', patience = 10):
    

    model.train()
    aupr_list = []

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr = learn_rate,
                                  weight_decay = 1e-3)
    
    
    

    roc_auc, pr_auc, valid_loss = eval_metric(valid, model, device, encoder)
    best_dev = math.inf
    best_epoc = math.inf
    model.train()
    steps = 0

    for epoch in tqdm(range(epoch)):
        
        loss = 0
        
        for batch_idx, batch_data in enumerate(train):
            
                X = batch_data[0]
                S = batch_data[1]

                label = batch_data[-1]
                optimizer.zero_grad()

                outputs = model(X,S).squeeze().to(device)

                train_loss = criterion(outputs, label)
                train_loss.backward()
                optimizer.step()
                loss += train_loss.item()
 
        print("Training Loss = ", loss)
        
        
    
        model.eval()
        

        roc_auc, pr_auc, valid_loss = eval_metric(valid, model, device, encoder)
        print("Dev = ", valid_loss)
        if best_dev > valid_loss:
            best_dev = valid_loss
            best_epoc = epoch
            torch.save(model, "saved_model/" + str(task) + ".p")
        if epoch - best_epoc == patience:
            model = torch.load("saved_model/" + str(task) + ".p")
            break
        model.train()
    model = torch.load("saved_model/" + str(task) + ".p")
    return model, aupr_list
   


   
    
