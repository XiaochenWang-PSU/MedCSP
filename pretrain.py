from sklearn import preprocessing
import numpy as np
from model import LSTM_Encoder, MLP_Encoder, MedCSP
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import sparse
from tqdm import tqdm
import pandas as pd
import pickle
import ast
from torch.nn.parallel import DistributedDataParallel as DDP
from info_nce import InfoNCE
import time
import torch.distributed as dist
from data import CXRDataset
from datetime import datetime
import time
from torch.cuda.amp import GradScaler, autocast
import open_clip

from open_clip import create_model_from_pretrained, get_tokenizer


BATCH = 128



device = torch.device("cuda:0")

ALPHA = 0.5
BETA = 0.2





base_image_dir = "base_image_dir"
base_text_dir = 'base_text_dir'
list_ids = pd.read_csv("list_ids")["subject_id"].apply(lambda x: 'p' + str(x)).tolist()
# Create dataset

# Create dataset
cxr_data = CXRDataset(base_image_dir, base_text_dir, admissions_csv, list_ids, False)
ehr_data = pickle.load(open("ehr_data.p", 'rb'))

tokenizer  = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

def collate_batch_ehr(batch_data):
    # Assuming icd and drug are already tensors or compatible formats
    icd = torch.stack([torch.tensor(i[0]) for i in batch_data]).float().to(device)
    drug = torch.stack([torch.tensor(i[1]) for i in batch_data]).float().to(device)
    
    # Convert COO sparse matrix to dense tensor for X and S
    X = torch.stack([torch.stack([torch.tensor(j.todense()) for j in k[2]], dim=0) for k in batch_data]).float().to(device)
    S = torch.stack([torch.tensor(k[3][0].todense()) for k in batch_data]).float().to(device)
    text = tokenizer([i[4] for i in batch_data]).to(device)
    pat = [i[5] for i in batch_data]
    hist = torch.tensor([i[6] for i in batch_data]).float().to(device)
    time = torch.tensor([i[7] for i in batch_data]).to(device)

    return [icd, drug, X, S, text, pat, hist, time]

def collate_batch_cxr(batch_data):

    image = torch.stack([i[0] for i in batch_data]).squeeze().to(device)
    text = tokenizer([i[1] for i in batch_data]).to(device)
    pat = [int(i[2]) for i in batch_data]
    hist = torch.stack([i[3] for i in batch_data]).float().to(device)
    time = torch.tensor([i[4] for i in batch_data]).to(device)
    return [image,text, pat, hist, time]

def compare_lists_to_tensor(list1, list2):
    comparison_matrix = [[int(x == y) for y in list2] for x in list1]
    tensor_result = torch.tensor(comparison_matrix)
    return tensor_result


model = MedCSP().to(device)

criterion = InfoNCE()

model.train()
scaler = GradScaler()
step = 0
start_t = time.time()
a_losses = 0
p_losses = 0
h_losses = 0

optimizer = torch.optim.AdamW(model.parameters(),
                          lr = 1e-5, weight_decay = 1e-8)

for epoch in tqdm(range(10)):
    


    torch.autograd.set_detect_anomaly(True)
    
    ehr_dataloader = iter(DataLoader(ehr_data, batch_size = BATCH, shuffle = True, collate_fn = collate_batch_ehr))
    cxr_dataloader = DataLoader(cxr_data, batch_size = BATCH, shuffle = True, collate_fn = collate_batch_cxr)
    print("start new epoch")
    for batch_idx, cxr in enumerate(cxr_dataloader):

        optimizer.zero_grad()
        image = cxr[0]
        cxr_text = cxr[1]
        cxr_pat = cxr[2]
        cxr_hist = cxr[3]
        cxr_time = cxr[4]
        try:

            icd, drug, ts, demo, ehr_text, ehr_pat, ehr_hist, ehr_time = next(ehr_dataloader)

        except:
            
            ehr_dataloader = iter(DataLoader(ehr_data, batch_size = BATCH, shuffle = True,  collate_fn = collate_batch_ehr))
            icd, drug,  ts, demo, ehr_text, ehr_pat, ehr_hist, ehr_time  = next(ehr_dataloader)
        if icd.shape[0] != image.shape[0]:
            continue
        

        pat_sim = compare_lists_to_tensor(ehr_pat, cxr_pat).float().to(device)
        time_dif = torch.abs(ehr_time - torch.t(cxr_time)).half().to(device) * pat_sim
        time_dif = time_dif#.half()
        his_sim = torch.matmul(ehr_hist, torch.t(cxr_hist)) / (torch.norm(ehr_hist, dim = 1) * torch.norm(cxr_hist, dim = 1))
        his_sim = his_sim#.half()
        

        icd = model(icd, modality = "icd")
        drug = model(drug, modality = "drug")
        demo = model(demo, modality = "demo")
        ts = model(ts, modality = "time_series")
        image = model(image, modality = "image")
        ehr_text = model(ehr_text, modality = "text")
        cxr_text = model(cxr_text, modality = "text")


        
        cxr_record = (image + cxr_text) /2
        ehr_record = (ts + demo + icd + drug + ehr_text) /5
        
        a_loss = (criterion(image, cxr_record) + criterion(cxr_text, cxr_record))/2 + (criterion(icd, ehr_record) + criterion(drug, ehr_record) + criterion(demo, ehr_record) + criterion(ts, ehr_record) + criterion(ehr_text, ehr_record))/5
        p_loss = ALPHA * criterion(ehr_record, cxr_record, labels = pat_sim, mask = time_dif)
        h_loss = BETA * criterion(ehr_record, cxr_record, labels = his_sim)

        
        

        loss = a_loss + h_loss + p_loss


        loss.backward()



        a_losses += a_loss.item()
        p_losses += p_loss.item()
        h_losses += h_loss.item()


        optimizer.step()
        step += 1
        if step % 1000 == 0:

            print(step)
            print(a_losses/500)

            a_losses = 0
            # p_losses = 0
            # h_losses = 0

        

    torch.save(model, "model.pt")


        

    