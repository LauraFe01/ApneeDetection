import os, sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from networks import Siamese_CRNN
from losses import ContrastiveLoss
from dataset import ApneaDataset
import gc

#TODO: PRIMA DI LANCIARE IL TRAINING CON PIU PAZIENTI ED EPOCHE RISOLVI IL BUG A LINEA 71

def start_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(log_dir = os.path.join(logs_path, 'new_exp_P' + str(P_n), 'tensorboard'))

    # define model
    model = Siamese_CRNN()
    model.to(device)

    # loss and optimizer
    criterion = ContrastiveLoss(margin = 2.0).to(device)
    # optimizer = optim.Adam(model.parameters(), lr = 5e-4, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2)
    optimizer = optim.SGD(model.parameters(), lr = 5e-3)

    # defining train dataset and dataloader    
    a_files = os.listdir(os.path.join(data_path, 'apnea'))
    a_lbls = [1]*len(a_files) #crea una lista di 1 di lunghezza a_files
    na_files = os.listdir(os.path.join(data_path, 'nonapnea'))
    na_lbls = [0]*len(na_files) #crea una lista di 0 di lunghezza na_files
    a_items = list(zip(a_files,a_lbls)) #associa ad ogni file l'etichetta corrispondente
    na_items = list(zip(na_files,na_lbls))
    print(f'Apnea items: {len(a_items)}')
    print(f'Non-apnea items: {len(na_items)}')
    train_set = ApneaDataset(data_path, a_items, na_items) #serve per creare coppie casuali di diversi tipi di combinazioni apnea non apnea
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    print(f'Start experiment')
    print(f'Found device: {device}\n')

    start_tstamp = datetime.now()

    model.train()

    training_loss = 0.0
    best_loss = 1e6

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}/{num_epochs}')

        # Training loop
        print(f'train...')
        for i, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            S_1, S_2, lbl_1, lbl_2 = batch
            E_1, E_2 = model(S_1.to(device), S_2.to(device))
            # dissim is the "dissimilarity" label: 1 (True) = different inputs, 0 (False) = same input class
            # it is computed with XOR       
            dissim = lbl_1 ^ lbl_2 #torna 0 se sono uguali 1 se sono diversi
            loss = criterion(E_1, E_2, dissim.to(device)) #loss
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if i%10 == 0: #ogni 10 elementi
                if training_loss < best_loss:
                    torch.save(model.state_dict(), os.path.join(logs_path, 'new_exp_P' + str(P_n), 'checkpoints', 'best_model.pth')) #salva il modello
                    best_loss = training_loss
                writer.add_scalar("train_loss", training_loss/10, i)     ####BUG: MODIFICARE L'INDICE i ALTRIMENTI COSI RIPARTE SEMPRE DA 0 E IL GRAFICO DELLA LOSS VIENE MALE
                writer.flush()
                training_loss = 0.0

    end_tstamp = datetime.now()
    writer.close()
    print(f'Training started at {start_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}, Training finished at {end_tstamp.strftime("%m/%d/%Y - %H:%M:%S")}')


if __name__ == "__main__":
    num_epochs = 1
    patients= [1108] # Patient number without leading zeros  
    
    logs_path = "/home/adanna/Codice/apnea_detection_v3/logs/"
    
    
    for P_n in patients:
        print(f'Patient: {P_n}')
        data_path = "/disks/disk1/adanna/MELSP_6S/P" + str(P_n)
        """         if not os.path.exists(os.path.join(logs_path, 'new_exp_P' + str(P_n))):
            os.makedirs(os.path.join(logs_path, 'new_exp_P' + str(P_n), 'checkpoints'))
            os.makedirs(os.path.join(logs_path, 'new_exp_P' + str(P_n), 'tensorboard'))
        else:
            raise Exception(f'Log folder for patient {str(P_n)} already exists!')
        """
        start_train()

        torch.cuda.empty_cache()
        gc.collect()