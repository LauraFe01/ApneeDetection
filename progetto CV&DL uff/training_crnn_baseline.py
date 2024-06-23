import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import CoTeachingDataset  # Assicurati che questo import sia corretto
from networks import CRNN_2
from torchmetrics.classification import Precision, Recall, BinaryF1Score
import gc
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import pandas as pd

# Parametri di addestramento
batch_size = 64
epochs = 150
learning_rate = 0.01
print_freq = 1
logs_path = "/home/adanna/Codice/apnea_detection_v3/model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Elenco dei pazienti con cui lavorare
training_patients = ["P1097","P1000", "P1006", "P1008", "P1010", "P1014", "P1016", "P1018", "P1020", "P1022","P1028","P1037", "P1039", "P1041", "P1043", "P1057", "P1059", "P1069",  "P1071","P1073"]  # 65%
validation_patients = ["P1108", "P1026", "P1084"]

# Crea un SummaryWriter per TensorBoard
writer = SummaryWriter(log_dir=os.path.join(logs_path, 'tensorboard'))

data_path = "/disks/disk1/adanna/MELSP_6S"

dataset_training = CoTeachingDataset(data_path, training_patients)
dataset_validation = CoTeachingDataset(data_path, validation_patients, val=True)

print(f'Dimensione dataset di training: {len(dataset_training)}')
print(f'Dimensione dataset di validazione: {len(dataset_validation)}')

train_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_validation, batch_size=1, shuffle=True)

# Inizializza il modello

model = CRNN_2().to(device)

# Definisci l'ottimizzatore
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Metriche per la valutazione
precision_metric = Precision(task='binary', average='micro').to(device)
recall_metric = Recall(task='binary', average='micro').to(device)
f1_metric = BinaryF1Score().to(device)

def training(train_loader, epoch, model, optimizer, best_loss, best_f1, writer):
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        target = target.float().unsqueeze(1)
        
        # Forward pass
        output = model(data)
        
        # Calcola la loss
        loss = F.binary_cross_entropy(output, target)
        
        # Backpropagation e ottimizzazione
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        # Stampa le informazioni
        if batch_idx % print_freq == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item()}')
    
    avg_loss = total_loss / total_batches
    scheduler.step(avg_loss)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            label = label.float().unsqueeze(1)
            prediction = model(data)
            all_predictions.append(prediction)
            all_labels.append(label)
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    precision_metric(all_predictions, all_labels)
    recall_metric(all_predictions, all_labels)
    f1_metric(all_predictions, all_labels)
    
    avg_precision = precision_metric.compute().item()
    avg_recall = recall_metric.compute().item()
    avg_f1 = f1_metric.compute().item()
    
    print(f'Precision: {avg_precision}, Recall: {avg_recall}, F1: {avg_f1}')
    torch.save(model.state_dict(), os.path.join(logs_path, 'checkpoints', f'baseline_model_{epoch}.pth'))
    
    # Aggiorna il miglior modello
    if avg_f1 > best_f1:
        best_loss = avg_loss
        best_f1 = avg_f1
        torch.save(model.state_dict(), os.path.join(logs_path, 'checkpoints', f'baseline_best_model_{epoch}.pth'))
        print(f'Best model saved with loss {best_loss} and F1 {best_f1}')
    
    # Aggiorna i risultati su TensorBoard
    writer.add_scalar("precision", avg_precision, epoch)
    writer.add_scalar("recall", avg_recall, epoch)
    writer.add_scalar("f1", avg_f1, epoch)
    writer.add_scalar("train_loss", avg_loss, epoch)
    
    # Salva le metriche in un file Excel
    data = {
        'epoch': [epoch],
        'precision': [avg_precision],
        'recall': [avg_recall],
        'f1': [avg_f1],
        'train_loss': [avg_loss]
    }

    df = pd.DataFrame(data)
    
    if not os.path.isfile('training_crnn.xlsx'):
        df.to_excel('training_crnn.xlsx', index=False)
    else:
        df_existing = pd.read_excel('training_crnn.xlsx')
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_excel('training_crnn.xlsx', index=False)
    
    writer.flush()
    
    return best_loss, best_f1

def main():
    best_loss = 1e6
    best_f1 = 0.0
    for epoch in range(epochs):
        best_loss, best_f1 = training(train_loader, epoch, model, optimizer, best_loss, best_f1, writer)
        print(f'Epoch [{epoch+1}/{epochs}] best_loss: {best_loss} best_f1: {best_f1}')
    
    writer.close()

if __name__ == "__main__":
    main()
   