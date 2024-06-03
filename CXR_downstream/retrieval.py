import argparse
import os
import yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score, precision_score, recall_score
from transformers import CLIPProcessor, CLIPModel
from datasets import RSNA, Chestxray14_Dataset
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from PIL import Image
from sklearn import preprocessing
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,precision_score,recall_score
from tqdm import tqdm
import pandas as pd
import pickle
import ast
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import torch.distributed as dist
from datetime import datetime
import time
from torch.cuda.amp import GradScaler, autocast
# from open_clip import create_model_from_pretrained
from datasets import MIMIC, OPENI
from medclip import MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT

device = torch.device("cuda:0")

def collate_fn_bio(data):
    images = [s['image'] for s in data]
    images_processed = []
    length = []
    #print(images)
    for image_list in images:
        length.append(len(image_list))
        for image in image_list:
            images_processed.append(preprocess_bio(image))
    images = torch.stack(images_processed)

    texts = [s['text'] for s in data]
    texts = tokenizer_bio(texts, context_length=context_length)
    return {
        'image': images,
        'text': texts,
        'length': length    
    }

def collate_fn_medclip(data):
        images = [s['image'] for s in data]
        images_processed = []
        length = []
        #print(images)
        for image_list in images:
            length.append(len(image_list))
            for image in image_list:
                images_processed.append(image)

        texts = [s['text'] for s in data]
        inputs = preprocess_mediclip(
            text=texts, 
            images=images_processed, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=context_length
            )
        return {
            'image': inputs,
            'text': texts,
            'length': length    
        }

def collate_fn_clip(data):
        images = [s['image'] for s in data]
        images_processed = []
        length = []
        #print(images)
        for image_list in images:
            length.append(len(image_list))
            for image in image_list:
                images_processed.append(image)

        texts = [s['text'] for s in data]
        inputs = preprocess_clip(
            text=texts, 
            images=images_processed, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=context_length
            )
        return {
            'image': inputs,
            'text': texts,
            'length': length    
        }
# text image-text retrieval
def test(args):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    model.eval()
    test_dataloader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            collate_fn=collate_fn,
            drop_last=False,
        ) 
    # initialize the ground truth and output tensor
    image_embeddings = torch.FloatTensor()
    text_embeddings = torch.FloatTensor()

    print("Start testing")
    model.eval()
    length_list = []
    for i, sample in tqdm(enumerate(test_dataloader)):
        images = sample['image'].to(device)
        texts = sample['text'].to(device) if not isinstance(sample['text'], list) else sample['text']
        # if isinstance(sample['text'], list):
            # print(len(texts))
        length = sample['length']
        #print(sample['length'])
        with torch.no_grad():
            image_embedding, text_embedding =model(images, texts)
            image_embedding = image_embedding.detach().float().cpu()
            text_embedding = text_embedding.detach().float().cpu()
            image_embeddings = torch.cat((image_embeddings, image_embedding), 0)
            text_embeddings = torch.cat((text_embeddings, text_embedding), 0)
            length_list.extend(length)
    #build the image-text retrieval index
    print("Building image-text retrieval index")
    text_size = text_embeddings.shape[0]
    image_size = image_embeddings.shape[0]
    text_image_ids = []
    count_images = 0
    for i in range(text_size):
        c_image_size = length_list[i]
        target_image_ids = []
        for image_id in range(c_image_size):
            target_image_ids.append(count_images + image_id)
        text_image_ids.append(target_image_ids)
        count_images += c_image_size
    #calculate the text-image retrieval precision at k and recall at k
    print("Calculating text-image retrieval precision at k")
    text_embeddings = text_embeddings.numpy()
    image_embeddings = image_embeddings.numpy()
    #nomalize the embeddings based on mod
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    similarity_all = np.dot(text_embeddings, image_embeddings.T)
    precision_at_k = []
    recall_at_k = []
    print(text_size, image_size)
    print(text_image_ids)
    for k in [1, 5, 10, 20, 50, 100]:
        precisions = []
        recalls = []
        for i in range(text_size):
            similarity = similarity_all[i]
            #calculate the precision at k
            topk_image_ids = np.argsort(-similarity)[:k]
            target_image_ids = text_image_ids[i]
            precision = len(set(topk_image_ids) & set(target_image_ids)) / k
            precisions.append(precision)
            #calculate the recall at k
            recall = len(set(topk_image_ids) & set(target_image_ids)) / len(target_image_ids)
            recalls.append(recall)
        precision_at_k.append(np.mean(precisions))
        recall_at_k.append(np.mean(recalls))
    print("precision_at_k: ", precision_at_k)
    print("recall_at_k: ", recall_at_k)
    #save results
    with open('./data/retrieval/{name}-{data}.txt'.format(name=args.model_path.split('/')[-1], data='MIMIC'), 'w') as f:
        f.write("precision_at_k: " + str(precision_at_k) + '\n')
        f.write("recall_at_k: " + str(recall_at_k) + '\n')

class BiomedCLIP(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base
    def forward(self, images, texts):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)
        return image_features, text_features

class OriginalCLIP(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base
    def forward(self, inputs, texts):
        image_features =  self.model.get_image_features(pixel_values=inputs['pixel_values'])
        text_features = self.model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return image_features, text_features
    
class MedCLIP(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base
    def forward(self, inputs, texts):
       outputs = self.model(**inputs)
       image_features = outputs['img_embeds']
       text_features = outputs['text_embeds']
       return image_features, text_features       
    
def load_model(model_path):
    print("loading model")
    if model_path == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224':
        model_base, _ = create_model_from_pretrained(args.model_path)
        model = BiomedCLIP(model_base)
        my_collate_fn = collate_fn_bio
    elif model_path == 'openai/clip-vit-large-patch14':
        model_base = CLIPModel.from_pretrained(args.model_path)
        model = OriginalCLIP(model_base)
        my_collate_fn = collate_fn_clip
    elif model_path == 'MedClip':
        model_base = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model_base.from_pretrained()
        model = MedCLIP(model_base)
        my_collate_fn = collate_fn_medclip
    elif model_path == 'csp_clip':
        model_base = torch.load("/path/to/model", map_location = 'cpu').clip
        model = BiomedCLIP(model_base)
        my_collate_fn = collate_fn_bio
    elif model_path == "flaviagiammarino/pubmed-clip-vit-base-patch32":
        model_base = CLIPModel.from_pretrained(args.model_path)
        model = OriginalCLIP(model_base)
        my_collate_fn = collate_fn_clip
    return model, my_collate_fn

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='cxr_clip')
    
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    context_length = 512
    device = torch.device("cuda:0")
   

    id_list = pickle.load(open("retrieval_ids.p", 'rb'))
    
    test_dataset =  MIMIC('/MIMIC-cxr/reports.json', '/physionet.org/files/mimic-cxr-jpg/2.0.0/files', id_list)
    test_dataset =  OPENI('/openI/texts/ecgen-radiology/', '/openI/images')
    if args.model_path == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' or args.model_path == 'csp_clip':
        tokenizer_bio = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        _, preprocess_bio = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        del _
    elif args.model_path == 'openai/clip-vit-large-patch14':
        preprocess_clip = CLIPProcessor.from_pretrained(args.model_path)
        context_length = 77
    elif args.model_path == 'MedClip':  
        preprocess_mediclip = MedCLIPProcessor()
    elif args.model_path == "flaviagiammarino/pubmed-clip-vit-base-patch32":  
        preprocess_clip = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
        context_length = 77
    elif args.model_path == 'cxr_clip':
        tokenizer_bio = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        _, preprocess_bio = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
    model, collate_fn = load_model(args.model_path)
    model.to(device)
    
    
    test(args)
