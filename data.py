import pandas as pd
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import pickle
from datetime import datetime   


class CXRDataset(Dataset):
    def __init__(self, base_image_dir, base_text_dir, admissions_csv, list_ids, transform=None):
        self.data = []
        self.transform = transform
        admissions = pd.read_csv(admissions_csv)
        study_time_map = {str(row['study_id']): datetime.strptime(str(row['StudyDate']), '%Y%m%d').timestamp()/86400 for index, row in admissions.iterrows()}
        pat = pickle.load(open("patient_history.p", 'rb'))
        pat, hist = pat["subject_id"].tolist(), pat["icd_union"].tolist()
        self.dic = dict(zip(pat, hist))
        # Loop through directories p10 to p19 and patient IDs
        for i in range(10, 20):
            image_dir = os.path.join(base_image_dir, f"p{i}")
            text_dir = os.path.join(base_text_dir, f"p{i}")
            
            for patient_id in list_ids:
                # Find all study folders for the patient
                study_folders = glob.glob(os.path.join(image_dir, patient_id, '*'))
                # print(patient_id)
                for study_folder in study_folders:
                    # Find all image files in the study folder
                    # print(os.path.join(study_folder, '*.pt'))
                    study_id = os.path.basename(study_folder)
                    image_paths = glob.glob(os.path.join(study_folder, '*.pt'))
                    # Corresponding text file path
                    # text_path = os.path.join(text_dir, patient_id, os.path.basename(study_folder) + '.pt')
                    text_path = os.path.join(text_dir, patient_id, os.path.basename(study_folder) + '.txt')
                    study_time = study_time_map.get(study_id.replace("s", ''))
                    
                    # print("non exist")
                    if not image_paths or not os.path.exists(text_path):
                        continue
                    # print("exist")
                    with open(text_path, 'r') as file:
                        text = file.read()

                    for image_path in image_paths:
                        self.data.append((image_path, text, patient_id, study_time))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, text, patient_id, study_time = self.data[idx]
        image = torch.load(image_path)
        if self.transform:
            image = self.transform(image)
        return image, text, patient_id.replace('p', ''), torch.tensor(self.dic[int(patient_id.replace('p', ''))]), study_time
