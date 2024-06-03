import csv
import json
import logging
import os
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import pickle
import pydicom

class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path):
        data_info = pd.read_csv(csv_path)
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])
        
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])
            

    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        class_label = self.class_list[index]

        return {
            "image": image_path,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class RSNA(Dataset):
    def __init__(self, path, process):
        csv_path = os.path.join(path, 'stage_2_train_labels.csv')
        data_info = pd.read_csv(csv_path)
        # self.img_path_list = np.asarray(data_info.iloc[:,0])
        # self.class_list = np.asarray(data_info.iloc[:,3:])
        self.path = path
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,5])
        self.process = process
        self.label_description = ['A clear and healthy lung X-ray showing normal lung tissue without any signs of infection, fluid, or abnormalities. The airways are unobstructed, and the lung fields appear pristine, with visible, well-defined lung markings. No evidence of pneumonia, consolidation, or other lung diseases.','A lung X-ray with signs of pneumonia, showing areas of consolidation or infection. There may be visible opacities, fluid, or infiltrates typically associated with pneumonia. The lung fields show signs of distress, with possible obscuring of the normal lung markings due to the presence of the disease.']
            

    def __getitem__(self, index):
        image_path = self.img_path_list[index]+'.dcm'
        class_label = self.class_list[index]
        ds = pydicom.dcmread(os.path.join(self.path, "stage_2_train_images", image_path))
        #covert to Image
        image_raw = Image.fromarray(ds.pixel_array)
        image = self.process(image_raw)
        return {
            "image": image,
            "image_raw": image_raw,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
class OPENI(Dataset):
    def __init__(self, text_path, image_path):
        text_files = os.listdir(text_path)
        self.img_path_list = []
        self.report_list = []
        for text_file in tqdm(text_files):
            text_path_patient_xml = os.path.join(text_path, text_file)
            #read the xml file and find the "COMPARISON" and "INDICATION" sections using beautifulsoup
            with open(text_path_patient_xml, 'r', encoding='utf-8') as f:
                xml = f.read()
            soup = BeautifulSoup(xml, 'xml')
            #print(soup.MedlineCitation)
            comparison = soup.find('AbstractText', Label='COMPARISON')
            indication = soup.find('AbstractText', Label='INDICATION')
            findings = soup.find('AbstractText', Label='FINDINGS')
            impression = soup.find('AbstractText', Label='IMPRESSION')
            report = 'COMPARIOSN: '+comparison.get_text()+'\n'+'INDICATION: '+indication.get_text()+'\n'+'FINDINGS: '+findings.get_text()+'\n'+'IMPRESSION: '+impression.get_text()+'\n'    
            #print(report)
            #find the associated images
            image_ids = soup.find_all('parentImage')
            paths = []
            for image_id in image_ids:
                image_path_patient = os.path.join(image_path, image_id['id']+'.png')
                #print(image_path_patient)
                paths.append(image_path_patient)
            if len(paths) == 0:
                print('skip', text_file)
            else:
                self.report_list.append(report)
                self.img_path_list.append(paths)


    def __getitem__(self, index):
        image_paths = self.img_path_list[index]
        report = self.report_list[index]
        image_raw = []
        for image_path in image_paths:
            image_raw.append(Image.open(image_path).convert('RGB'))
        return {
            "image": image_raw,
            "text": report
            }
    def __len__(self):
        return len(self.img_path_list)
    
class MIMIC(Dataset):
    def __init__(self, text_path, image_path, id_list):
        reports = json.load(open(text_path, 'r', encoding='utf-8'))
        self.img_path_list = []
        self.report_list = []
        for id_patient in id_list:
            img_path_patient = os.path.join(image_path, id_patient[0:3], id_patient[0:9], id_patient[9:])
            image_names = os.listdir(img_path_patient)
            self.report_list.append(reports[id_patient])
            paths = []
            for image_name in image_names:
                if 'index.html' == image_name:
                    continue
                paths.append(os.path.join(img_path_patient, image_name))
            self.img_path_list.append(paths)
    def __getitem__(self, index):
        image_paths = self.img_path_list[index]
        report = self.report_list[index]
        image_raw = []
        for image_path in image_paths:
            image_raw.append(Image.open(image_path).convert('RGB'))
        return {
            "image": image_raw,
            "text": report
            }
    def __len__(self):
        return len(self.img_path_list)

    
class COVID(Dataset):
    def __init__(self, path, process, sub_type):
        record_txt = os.path.join(path, '{}.txt'.format(sub_type))
        # read the record txt and get the image path and label
        with open(record_txt, 'r') as f:
            lines = f.readlines()
        self.img_path_list = []
        self.class_list = []
        for line in lines:
            line = line.strip().split()
            self.img_path_list.append(os.path.join(path, sub_type, line[1]))
            self.class_list.append(int(line[2]=='positive'))
        print(sum(self.class_list)/len(self.class_list))
        self.process = process
        # shuffle the data uisng a fixed seed
        self.img_path_list = np.asarray(self.img_path_list)
        self.class_list = np.asarray(self.class_list)
        if sub_type == 'train':
            if os.path.exists(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type))):
                with open(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type)), 'rb') as f:
                    idx = pickle.load(f)
                
            else:
                np.random.seed(42)
                idx = np.arange(len(self.img_path_list))
                np.random.shuffle(idx)
                print(idx[0:20])
                with open(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type)), 'wb') as f:
                    pickle.dump(idx, f)

            self.img_path_list = self.img_path_list[idx]
            self.class_list = self.class_list[idx]
        
            

    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        class_label = self.class_list[index]
        image_raw = Image.open(image_path).convert('RGB')
        image = self.process(image_raw)
        return {
            "image": image,
            "image_raw": image_raw,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)
 
class COVID2(Dataset):
    def __init__(self, path, process):
        self.img_path_list = []
        self.class_list = []
        self.process = process
        sub_type = 'COVID'
        for image_name in os.listdir(os.path.join(path, sub_type, 'images')):
            self.img_path_list.append(os.path.join(path, sub_type, 'images', image_name))
            self.class_list.append(1)
        sub_type = 'Normal'
        for image_name in os.listdir(os.path.join(path, sub_type, 'images')):
            self.img_path_list.append(os.path.join(path, sub_type, 'images', image_name))
            self.class_list.append(0)
        # shuffle the data uisng a fixed seed
        self.img_path_list = np.asarray(self.img_path_list)
        self.class_list = np.asarray(self.class_list)
        if os.path.exists(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type))):
            with open(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type)), 'rb') as f:
                idx = pickle.load(f)
            
        else:
            np.random.seed(42)
            idx = np.arange(len(self.img_path_list))
            np.random.shuffle(idx)
            print(idx[0:20])
            with open(os.path.join(path, 'seed-covid-{}.pkl'.format(sub_type)), 'wb') as f:
                pickle.dump(idx, f)

        self.img_path_list = self.img_path_list[idx]
        self.class_list = self.class_list[idx]
        self.label_description = ['A clear and healthy lung X-ray showing normal lung tissue without any signs of infection, fluid, or abnormalities. The airways are unobstructed, and the lung fields appear pristine, with visible, well-defined lung markings. No evidence of pneumonia, consolidation, or other lung diseases.','A lung X-ray or CT scan showing characteristic features of COVID-19. The image may exhibit bilateral and peripheral ground-glass opacities, consolidation, or a combination of both. These opacities often have a rounded morphology and may be associated with other features like crazy paving or a halo sign. The lung fields often show signs of viral pneumonia, distinct from bacterial infections or normal lung tissue.']
    
    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        class_label = self.class_list[index]
        image_raw = Image.open(image_path).convert('RGB')
        image = self.process(image_raw)
        return {
            "image": image,
            "image_raw": image_raw,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class CheXpert(Dataset):

    def __init__(self, base_path, process, policy='diff', target_class=0):
        self.img_path_list = []
        self.class_list = []
        self.process = process
        self.batch_folders = ['CheXpert-v1.0 batch 2 (train 1)', 'CheXpert-v1.0 batch 3 (train 2)', 'CheXpert-v1.0 batch 4 (train 3)']
        self.base_path = base_path
        csvpath = os.path.join(base_path, 'train_cheXbert.csv')
        with open(csvpath, 'r') as f:
            reader = csv.reader(f)
            idx = [6, 9, 10, 12, 14]
            for lid, line in enumerate(reader):
                if lid == 0:
                    # print the head of idx
                    print([line[i] for i in idx])
                    print('target class:', line[idx[target_class]])
                    continue
                npline = np.array(line)
                label = list(npline[idx])
                if label[target_class]:
                    a = float(label[target_class])
                    if a == 1:
                        label[target_class] = 1
                    elif a == -1:
                        if policy == 'diff':
                            if target_class in [1, 3, 4]:  # Atelectasis, Edema, Pleural Effusion
                                label[target_class] = 1                    # U-Ones
                            elif target_class in [0, 2]:          # Cardiomegaly, Consolidation
                                label[target_class] = 0                    # U-Zeroes
                            else:
                                label[target_class] = 0                   # U-Zeroes
                        elif policy == 'ones':              # All U-Ones
                            label[target_class] = 1
                        else:
                            label[target_class] = 0                    # All U-Zeroes
                    else:
                        label[target_class] = 0
                    correct_path = self.find_correct_path(line[0].split("CheXpert-v1.0/")[-1])
                    self.img_path_list.append(correct_path)
                    # self.img_path_list.append(os.path.join(base_path, line[0].split("CheXpert-v1.0/")[-1]))
                    self.class_list.append(label[target_class]) 
        # use chatgpt to create class description prompt
        self.label_description = ['CT image of a healthy lung showing normal heart size within the thoracic cavity, with the heart occupying less than half the width of the chest. The lung fields appear clear with no signs of fluid accumulation, congestion, or enlarged heart silhouette. The pulmonary vessels and airways are normal in appearance without any abnormalities. This image represents a typical example of a lung without Cardiomegaly, indicating no enlargement of the heart and no underlying heart disease.', 'CT image of a lung demonstrating Cardiomegaly, characterized by an enlarged heart silhouette that occupies more than half the width of the thoracic cavity. The enlargement is evident with a noticeable increase in the size of the heart shadow, potentially accompanied by signs of fluid accumulation in the lung fields, indicative of congestive heart failure. The pulmonary vessels may appear engorged, and there may be evidence of pulmonary congestion. This image serves as a clear example of Cardiomegaly, reflecting the presence of an underlying heart condition.']
        # count the number of positive and negative samples
        print(sum(self.class_list)/len(self.class_list))
        print(len(self.class_list))
    def __getitem__(self, index):
        image_path = self.img_path_list[index]
        class_label = self.class_list[index]
        image_raw = Image.open(image_path).convert('RGB')
        image = self.process(image_raw)
        return {
            "image": image,
            "image_raw": image_raw,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

    def find_correct_path(self, relative_path):
        """Find the correct path for the file across batch folders."""
        for batch_folder in self.batch_folders:
            potential_path = os.path.join(self.base_path, batch_folder, relative_path)
            potential_path = potential_path.replace("/train", "")
            print(potential_path)
            if os.path.exists(potential_path):
                return potential_path  # Return the first found path
        return None  # Return None if the file is not found in any batch folders
import torch
from medclip import MedCLIPProcessor
def collate_fn(data):
    images = torch.stack([s['image'] for s in data])
    labels = torch.LongTensor([s['label'] for s in data])
    images_raw = [s['image_raw'] for s in data]
    return {
        'image': images,
        'image_raw': images_raw,
        'label': labels
    }
preprocess_mediclip = MedCLIPProcessor()
def collate_fn_mediclip(data):
    images = torch.stack([s['image'] for s in data])
    labels = torch.LongTensor([s['label'] for s in data])
    images_raw = [s['image_raw'] for s in data]
    inputs = preprocess_mediclip(
        text=["negative", 
            "positive"], 
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
