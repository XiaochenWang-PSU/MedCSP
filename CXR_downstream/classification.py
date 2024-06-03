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
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,precision_score,recall_score

from datasets import RSNA, Chestxray14_Dataset, collate_fn, COVID2, CheXpert
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from PIL import Image
from medclip import MedCLIPProcessor
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda:0")
preprocess_mediclip = MedCLIPProcessor()
preprocess_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
def dummy_preprocess(x):
    return x

def test(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    context_length = 256
    def collate_fn_mediclip(data):
        images = [s['image'] for s in data]
        labels = torch.LongTensor([s['label'] for s in data])
        images_raw = [s['image_raw'] for s in data]
        inputs = preprocess_mediclip(
            text=new_label_description, 
            images=images_raw, 
            return_tensors="pt", 
            padding=True
            )
        return {
            'image': images,
            'image_raw': images_raw,
            'label': labels,
            'inputs': inputs
        }
    
    def collate_fn_clip(data):
        images = [s['image'] for s in data]
        labels = torch.LongTensor([s['label'] for s in data])
        images_raw = [s['image_raw'] for s in data]
        inputs = preprocess_clip(
            text=new_label_description, 
            images=images_raw, 
            return_tensors="pt", 
            padding=True
            )
        return {
            'image': images,
            'image_raw': images_raw,
            'label': labels,
            'inputs': inputs
        }
    print("loading model")
    if args.model_path != 'MedClip' and args.model_path != 'openai/clip-vit-large-patch14' and args.model_path != "flaviagiammarino/pubmed-clip-vit-base-patch32": 
        if args.model_path == "csp_clip":
          _, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        else:                      
          _, preprocess = create_model_from_pretrained(args.model_path)
        del _
        my_collate_fn = collate_fn
    else:
        preprocess = dummy_preprocess
        if args.model_path == 'MedClip':
            my_collate_fn = collate_fn_mediclip
        else:
            my_collate_fn = collate_fn_clip
    model.eval()
    if 'RSNA' in args.test_data:
        DataClass = RSNA
    elif 'covid' in args.test_data:
        DataClass = COVID2
    elif 'chexpert' in args.test_data:
        DataClass = CheXpert
    test_dataset =  DataClass(args.test_data, preprocess) 
    
    print("Creating label description")
    label_description = test_dataset.label_description
    print("label_description: ", label_description)
    texts = tokenizer(label_description, context_length=context_length).to(device)
    # only keep the frist two sentences in the label description by using the .
    new_label_description = []
    for label in label_description:
        label = label.split('.')[0:2]
        label = '.'.join(label)
        new_label_description.append(label)
    print("new_label_description: ", new_label_description)

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
            sampler=None,
            collate_fn=my_collate_fn,
            drop_last=False,
        ) 
    # initialize the ground truth and output tensor
    gt = torch.LongTensor()
    pred = torch.FloatTensor()
    
    print("Start testing")
    model.eval()
    for i, sample in tqdm(enumerate(test_dataloader)):
        images = sample['image'].to(device) if args.model_path not in ['MedClip', 'openai/clip-vit-large-patch14', "flaviagiammarino/pubmed-clip-vit-base-patch32"] else sample['inputs'].to(device)
        labels = sample['label']
        gt = torch.cat((gt, labels), 0) 
        with torch.no_grad():
            # print(images, texts)
            logits = model(images, texts)
            pred = torch.cat((pred, logits.detach().cpu()), 0)
        if i%50==0:
            accuracy = accuracy_score(gt.numpy(), np.argmax(pred.numpy(), axis=1))
            print("Accuracy by far: ", accuracy)
    #compute accuracy, precision, recall, auc
    # gt [B] pred [B,C] whre gt is the index of the label, pred is the probability of each label
    pred = pred.numpy()
    gt = gt.numpy()
    pred_index = np.argmax(pred, axis=1)
    accuracy = accuracy_score(gt, pred_index)
    print("Accuracy: ", accuracy)
    #compute AUC
    pred = pred[:,1]
    auc = roc_auc_score(gt, pred)
    print("AUC: ", auc)
    #compute precision and recall score based on the definition
    print(gt, pred_index)
    precision = precision_score(gt, pred_index)
    print("Precision: ", precision)
    recall = recall_score(gt, pred_index)
    print("Recall: ", recall)


    #compute F1
    f1 = 2 * precision * recall / (precision + recall)
    print("F1: ", f1)
    #save the results
    model_name = args.model_path.split('/')[-1]
    with open('./data/zero-shot/{name}-{data}.txt'.format(name=model_name, data=DataClass.__name__), 'w') as f:
        f.write("Accuracy: " + str(accuracy) + '\n')
        f.write("AUC: " + str(auc) + '\n')
        f.write("Precision: " + str(precision) + '\n')
        f.write("Recall: " + str(recall) + '\n')
        f.write("F1: " + str(f1) + '\n')

class ImageClassificationModelBio(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base

    def forward(self, images, texts):
        image_features, text_features, logit_scale = self.model(images, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        return logits

class ImageClassificationModel(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base

    def forward(self, images, texts):
        outputs = self.model(**images)
        logits = outputs.logits_per_image
        # logits = outputs[-1]
        return logits
    
from medclip import MedCLIPProcessor
class ImageClassificationModelMedClip(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.model = model_base

    def forward(self, images, texts):
        outputs = self.model(**images)
        logits = outputs['logits']
#        outputs = self.model(images)
#        logits = outputs[-1]
        return logits

class ImageClassificationModelLLVA(nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.visual = model_base.vision_tower
        self.text = model_base
    def forward(self, images, texts):
        image_forward_outs = self.visual(images, output_hidden_states=True)
        select_hidden_state = image_forward_outs.hidden_states[-2]
        image_features = select_hidden_state[:, 1:]
        image_features = image_features.mean(dim=1)
        text_features = self.text(texts, output_hidden_states=True).hidden_states[-2][:, 1:]
        logits = (image_features @ text_features.t()).detach().softmax(dim=-1)
        return logits

def load_model(model_path):
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    if model_path == 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224':
        print("loading model")
        model_base, preprocess = create_model_from_pretrained(args.model_path)
        model = ImageClassificationModelBio(model_base)
        tokenizer = get_tokenizer(args.model_path)
        model.to(device)
    elif model_path == 'MedClip':
        from medclip import MedCLIPModel, MedCLIPVisionModelViT
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        print(model)
        print(model.text_model.model.embeddings)
        model.from_pretrained()
        model = ImageClassificationModelMedClip(model)
        model.to(device)
    elif model_path =='csp_clip':
        model_base, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model_base = torch.load("model", map_location = 'cpu').clip
        model = ImageClassificationModelBio(model_base)
        tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        model.to(device)
    elif model_path == "flaviagiammarino/pubmed-clip-vit-base-patch32":
        model_base = CLIPModel.from_pretrained(args.model_path)
        model = ImageClassificationModel(model_base)
        model.to(device)
        # my_collate_fn = collate_fn_clip
    else:
        print("loading model")
        model_base = CLIPModel.from_pretrained(args.model_path)
        model = ImageClassificationModel(model_base)
        model.to(device)

    return model, tokenizer

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', default='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    # parser.add_argument('--model_path', default="flaviagiammarino/pubmed-clip-vit-base-patch32")
    # parser.add_argument('--model_path', default='openai/clip-vit-large-patch14')
    parser.add_argument('--model_path', default='MedClip')
    parser.add_argument('--test_data', default='/chexpert/chexpertchestxrays-u20210408/')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)

    
    test(args)
