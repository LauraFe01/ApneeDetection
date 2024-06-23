import os, sys
import random
import numpy as np
from sklearn.utils import resample
import torch
from torch.utils.data import Dataset, DataLoader

class CoTeachingDataset(Dataset):
    def __init__(self, root_dir, patients, val=False):
        self.root_dir = root_dir
        self.classes = ["nonapnea", "apnea"]
        self.class_to_label = {class_name: i for i, class_name in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        self.patients=patients
        
        for patient in self.patients:
            patient_folder = os.path.join(root_dir, patient)
            #print(patient_folder)
        
            for class_name in self.classes:  # sekf.classes contiente tutte le classi presenti nella directory principale
                #print(class_name)
                class_dir = os.path.join(patient_folder, class_name)
                #print(class_dir)
                files = os.listdir(class_dir)
                #print(files)

                # Perform resampling if necessary
                if val == False: 
                    if class_name == "apnea":
                        if len(files) < len(os.listdir(os.path.join(patient_folder, "nonapnea"))):
                            files = resample(files, replace=True, n_samples=len(os.listdir(os.path.join(patient_folder, "nonapnea"))), random_state=42)
                            #print(files)
                            
                    if class_name == "nonapnea":
                        if len(files) < len(os.listdir(os.path.join(patient_folder, "apnea"))):
                            #print(f'len vecchia {len(files)}')
                            files = resample(files, replace=True, n_samples=len(os.listdir(os.path.join(patient_folder, "apnea"))), random_state=42)
                            #print(f'len nuova {len(files)}')
                    
                for file_name in files:
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    #print("file path:", self.file_paths)
                    self.labels.append(self.class_to_label[class_name])
                    #controllato viene assognato 0 a NON APNEA e 1 ad APNEA
                    
            
                    

    def __len__(self):
        return len(self.file_paths)  # numero totale degli elementi nel dataset

    def __getitem__(self, idx):
        """ if torch.is_tensor(idx):
            idx = idx.tolist() """

        file_path = self.file_paths[idx]
        data = torch.load(file_path)

        label = self.labels[idx]
        data = data.unsqueeze(0)

        return data, label  # ritorna il dato e l'etichetta corrispondente per un dato indice.

