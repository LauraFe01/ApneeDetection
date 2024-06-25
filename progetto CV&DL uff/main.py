import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import CoTeachingDataset  
from networks import CRNN_2 
from coteaching_loss import loss_coteaching 
from torchmetrics.classification import Precision, Recall, BinaryF1Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc
from datetime import datetime
import sys
import pandas as pd
from codecarbon import EmissionsTracker

# Parametri di addestramento
batch_size = 64
epochs = 150
num_gradual = 30  # Numero di epoche in cui si raggiunge il forget_rate
exponent = 1
learning_rate = 0.01
forget_rate = 0.3
print_freq = 1
logs_path = "/home/adanna/Codice/apnea_detection_v3/model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tracker = EmissionsTracker(output_dir='emission_reports')

# Elenco dei pazienti
training_patients = ["P1024", "P1045", "P1097","P1000", "P1006", "P1008", "P1010", "P1014", "P1016", "P1018", "P1020", "P1022", "P1028","P1037", "P1039", "P1041", "P1043", "P1057", "P1059", "P1069",  "P1071","P1073", "P1089"]  # 65% - 20 pazienti
validation_patients = ["P1108", "P1026", "P1084", "P1110", "P1093"] # 15% - 3 pazienti 
writer = SummaryWriter(log_dir=os.path.join(logs_path, 'tensorboard'))
a_files_total=[]
a_lbls_total=[]
na_files_total=[]
na_lbls_total=[]

data_path = f"/disks/disk1/adanna/MELSP_6S"

dataset_training = CoTeachingDataset(data_path, training_patients)
dataset_validation = CoTeachingDataset(data_path, validation_patients, val=True)

print(f'Dimensione dataset: {len(dataset_training)}')
print(f'Dimensione dataset: {len(dataset_validation)}')

train_loader = DataLoader(dataset_training, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_validation, batch_size=1, shuffle=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

#modello1 inizializzazione
seed_1 = 128
set_seed(seed_1)
model_1 = CRNN_2().to(device)

#modello2 inizializzazione
seed_2 = 12
set_seed(seed_2)
model_2 = CRNN_2().to(device)

# Definisci ottimizzatori
optimizer_1 = optim.Adam(model_1.parameters(), lr=learning_rate)
scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.1, patience=5, verbose=True)

optimizer_2 = optim.Adam(model_2.parameters(), lr=learning_rate)
scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.1, patience=5, verbose=True)

# Calcola il rate schedule
rate_schedule = np.ones(epochs) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)

# Metriche per la valutazione
precision_1_metric = Precision(task='binary', average='micro').to(device)
recall_1_metric = Recall(task='binary', average='micro').to(device)
f1_1_metric = BinaryF1Score().to(device)
precision_2_metric = Precision(task='binary', average='micro').to(device)
recall_2_metric = Recall(task='binary', average='micro').to(device)
f1_2_metric = BinaryF1Score().to(device)

def approximate_value(value, threshold):
    if value >= threshold:
        return 1
    else:
        return 0

def training(train_loader, epoch, model_1, optimizer_1, model_2, optimizer_2, best_loss, best_f1, writer):
    model_1.train()
    model_2.train()

    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_batches = 0

    print(f'Inizio epoca {epoch} con num_remember {rate_schedule[epoch]}')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        target = target.float().unsqueeze(1)   
    
        # Forward pass
        output_1 = model_1(data)
        output_2 = model_2(data)

        # Calcola la loss di Co-Teaching
        ind_1_update, ind_2_update = loss_coteaching(output_1, output_2, target, rate_schedule[epoch], batch_idx)

        # Ottimizza model_1
        optimizer_1.zero_grad()
        selected_data2 = data[ind_2_update].to(device)
        selected_target_data2 = target[ind_2_update].to(device) 
        output_1_selected = model_1(selected_data2)
        loss1_selected = F.binary_cross_entropy(output_1_selected, selected_target_data2)
        loss1_selected.backward()
        optimizer_1.step()

        # Ottimizza model_2
        optimizer_2.zero_grad()
        selected_data1 = data[ind_1_update].to(device)
        selected_target_data1 = target[ind_1_update].to(device) 
        output_2_selected = model_2(selected_data1)
        loss2_selected = F.binary_cross_entropy(output_2_selected, selected_target_data1)
        loss2_selected.backward()
        optimizer_2.step()


        total_loss_1 += loss1_selected.item()
        total_loss_2 += loss2_selected.item()
        total_batches += 1

        if batch_idx % print_freq == 0:
            print('Batch [{}/{}]\tLoss_1: {:.6f}\tLoss_2: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset), loss1_selected.item(), loss2_selected.item()))

    avg_loss_1 = total_loss_1 / total_batches
    avg_loss_2 = total_loss_2 / total_batches

    # Aggiorna gli scheduler con le medie delle perdite
    scheduler_1.step(avg_loss_1)
    scheduler_2.step(avg_loss_2)

    #valutazione al termine di ogni epoca su validation set
    model_1.eval()
    model_2.eval()
    all_predictions_1 = []
    all_predictions_2 = []
    all_labels = []
    
    for data, label in val_loader:
        data, label = data.to(device), label.to(device)
        label = label.float().unsqueeze(1)
        with torch.no_grad():
            prediction1 = model_1(data)
            prediction2 = model_2(data)
        
        all_predictions_1.append(prediction1)
        all_predictions_2.append(prediction2)
        all_labels.append(label)
    
    all_predictions_1 = torch.cat(all_predictions_1)
    all_predictions_2 = torch.cat(all_predictions_2)
    all_labels = torch.cat(all_labels)

    precision_1_metric(all_predictions_1, all_labels)
    recall_1_metric(all_predictions_1, all_labels)
    f1_1_metric(all_predictions_1, all_labels)
    precision_2_metric(all_predictions_2, all_labels)
    recall_2_metric(all_predictions_2, all_labels)
    f1_2_metric(all_predictions_2, all_labels)

    # Calcola le metriche medie
    avg_precision_1 = precision_1_metric.compute().item()
    avg_recall_1 = recall_1_metric.compute().item()
    avg_f1_1= f1_1_metric.compute().item()
    avg_precision_2 = precision_2_metric.compute().item()
    avg_recall_2 = recall_2_metric.compute().item()
    avg_f1_2= f1_2_metric.compute().item()
    print(f'Precision_1: {avg_precision_1}, Recall_1: {avg_recall_1}')
    print(f'Precision_1: {avg_precision_2}, Recall_2: {avg_recall_2}')
    print(f'F1_1: {avg_f1_1}, F1_2: {avg_f1_2}')

    # Aggiorna e sakva il miglior modello
    training_losses = [avg_loss_1, avg_loss_2]
    training_f1s = [avg_f1_1, avg_f1_2]
    for i in range(2):
        training_loss = training_losses[i]
        training_f1 = training_f1s[i]
        torch.save(model_1.state_dict(), os.path.join(logs_path, 'checkpoints', f'24_model1_{epoch}.pth'))
        torch.save(model_2.state_dict(), os.path.join(logs_path, 'checkpoints', f'24_model2_{epoch}.pth'))
        if  training_f1 > best_f1:
            if i == 0:
                model = model_1
            else:
                model = model_2
            best_loss = training_loss
            best_f1 = training_f1
            print(f'new_best_model_{best_loss}_{datetime.today().strftime("%Y-%m-%d")}.pth')
            torch.save(model.state_dict(), os.path.join(logs_path, 'checkpoints', f'24_best_model_{epoch}.pth'))
            print("best loss aggiornata")
        print("train_loss", training_loss, batch_idx) 
        print("best_loss: ", best_loss)
    print(f'Ending epoch {epoch} with num_remember {rate_schedule[epoch]}')

    # Aggiorna i risultati su TensorBoard
    writer.add_scalar("precision_1", avg_precision_1, epoch)
    writer.add_scalar("recall_1", avg_recall_1, epoch)
    writer.add_scalar("f1_1",avg_f1_1, epoch)
    writer.add_scalar("precision_2", avg_precision_2, epoch)
    writer.add_scalar("recall_2", avg_recall_2, epoch)
    writer.add_scalar("f1_2",avg_f1_2, epoch)
    writer.add_scalar("train_loss1", loss1_selected, epoch)
    writer.add_scalar("train_loss2", loss2_selected, epoch)
    
    #creazione file excel con report training
    data = {
        'epoch': [epoch],
        'precision_1': [avg_precision_1],
        'recall_1': [avg_recall_1],
        'f1_1': [avg_f1_1],
        'precision_2': [avg_precision_2],
        'recall_2': [avg_recall_2],
        'f1_2': [avg_f1_2],
        'train_loss1': [loss1_selected.item()],
        'train_loss2': [loss2_selected.item()]
    }

    # Crea un DataFrame
    df = pd.DataFrame(data)

    # Scrive il DataFrame su un file Excel
    if not os.path.isfile('new_training_metrics.xlsx'):
        df.to_excel('new_training_metrics.xlsx', index=False)
    else:
        df_existing = pd.read_excel('new_training_metrics.xlsx')
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_excel('new_training_metrics.xlsx', index=False)

    writer.flush()
    
    return best_loss, best_f1

def main():
    best_loss = 1e6
    best_f1 = 0.001
    tracker.start()
    for epoch in range(epochs):
        best_loss, best_f1 = training(train_loader, epoch, model_1, optimizer_1, model_2, optimizer_2, best_loss, best_f1, writer)
        print(f'Epoch [{epoch+1}/{epochs}] best_loss: {best_loss} best_fi: {best_f1}')

    writer.close()
    tracker.stop()



if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    gc.collect()
