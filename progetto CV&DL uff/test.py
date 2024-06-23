import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from networks import CRNN_2
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import librosa

wav_path = "/disks/disk1/adanna/PSG_Audio/EDF/" #file path alla cartella con i file audio
checkpoint_path = "/home/adanna/Codice/apnea_detection_v3/model/checkpoints" #file path dove è salvato il modello
meta_file = "/home/adanna/Codice/apnea_detection_v3/meta.csv" # file .csv con informazioni circa i dati a disposizione 
SR = 16000 # working sample rate

E_type = ['ObstructiveApnea', 'CentralApnea', 'MixedApnea', 'Hypopnea'] # tipi di apnea nella colonna type del file csv
context_s = 6 # secondi di contesto

patients = [1120, 1106, 1082, 1095] # 20% - 4 pazienti 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CRNN_2()

net.load_state_dict(torch.load(os.path.join(checkpoint_path, "1706_model1_149.pth"))) #carico il modello che voglio usare
net = net.to(device)

net.eval()
all_predictions=[]
true_label=[]

#USARE QUESTO PER IL TESTING

for P_n in patients:
    print(f'Processing patient number {P_n}')
    
    meta_df = pd.read_csv(meta_file, low_memory=False)
    apnea_df = meta_df.loc[(meta_df['Type'] == E_type[0])|(meta_df['Type'] == E_type[1])|(meta_df['Type'] == E_type[2])|(meta_df['Type'] == E_type[3])]
    apnea_df = apnea_df.loc[apnea_df["Patient_ID"] == P_n]

    print(apnea_df)
    wav_files = os.listdir(os.path.join(wav_path, f"0000{P_n}"))
    s_len = 0.0
    for wf in wav_files:
        print("wf", wf)
        d = librosa.get_duration(path=os.path.join(wav_path, f"0000{P_n}", wf))
        s_len += d
    print(f'duration: {s_len}')

    # Build label mask
    apneas_mask = torch.zeros(int(s_len * SR)).int() #tensore di zeri
    
    for idx, row in apnea_df.iterrows():
        start = int(row['Start'] * SR)
        dur = int(row['Duration'] * SR)
        apneas_mask[start:start + dur] = 1 #creazione di maschera corrispondente ai momenti di apnea segnati con 1
    
    apneas_mask_list = apneas_mask.tolist()

    # Concatenate 1-hour audio files into a single array
    y_block = np.zeros((len(wav_files), 3600 * SR))
    for w in range(len(wav_files)):
        fname = f"0000{P_n}-100507[00{w + 1}].wav"
        y, _ = librosa.load(os.path.join(wav_path, f"0000{P_n}", fname), sr=SR)
        y_block[w, 0:y.shape[0]] += y
    y_all = y_block.reshape(-1)

   
    step_size = int((context_s * SR))
    for n, c in enumerate(tqdm(range(0,apneas_mask.shape[0],step_size))):
        print(apneas_mask[c:c+(context_s*SR)])
        print("*************")
        
        if sum(apneas_mask[c:c+(context_s*SR)]) == 0: # IF NON APNEA
            true_label.append(0)
            y_chunk = y_all[c:c + (context_s * SR)]
            S2 = librosa.feature.melspectrogram(y=y_chunk, sr=SR, hop_length=185, n_mels=64)
            S2_dB = librosa.power_to_db(S2, ref=np.max)  # shape [64, 519]
            S2_dB = torch.from_numpy(S2_dB).float().to(device)
            
        elif sum(apneas_mask[c:c+(context_s*SR)]) == (context_s*SR):
            true_label.append(1)
            y_chunk = y_all[c:c + (context_s * SR)]
            S2 = librosa.feature.melspectrogram(y=y_chunk, sr=SR, hop_length=185, n_mels=64)
            S2_dB = librosa.power_to_db(S2, ref=np.max)  # shape [64, 519]
            S2_dB = torch.from_numpy(S2_dB).float().to(device)
        else:
            continue
        
        print("true_label",true_label)
        print(len(true_label))
        print("----------")

        # Perform inference with the model
        with torch.no_grad():
            S2_dB = S2_dB.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            #print(S2_dB)
            output = net(S2_dB)

        if output.item() < 0.5:
            print("output < 0.5",output.item())
            pred = 0
        else:
            print("output >= 0.5",output.item())
            pred = 1

        all_predictions.append(pred)
        print("all_predictions",all_predictions)
        print(len(all_predictions))
        
        print("---------")

precision = precision_score(true_label, all_predictions)
recall = recall_score(true_label, all_predictions)
f1 = f1_score(true_label, all_predictions)

print(f"true label after processing \n {true_label}")   
print(f"predicted label after processing \n {all_predictions}") 

print(f'Precision before processing: {precision:.4f}')
print(f'Recall before processing: {recall:.4f}')
print(f'F1 Score before processing: {f1:.4f}')

#se c'è un 1 preceduto e succeduto da almeno 2 zeri impostare 1 a 0
i = 0
while i < len(all_predictions):
    if all_predictions[i] == 1:
        count_zeros_before = 0
        j = i - 1
        while j >= 0 and all_predictions[j] == 0:
            count_zeros_before += 1
            j -= 1
        count_zeros_after = 0
        k = i + 1
        while k < len(all_predictions) and all_predictions[k] == 0:
            count_zeros_after += 1
            k += 1

        if count_zeros_before >= 2 and count_zeros_after >= 2:
            all_predictions[i] = 0
            i += 3
        else:
            i += 1
    else:
        i += 1

i = 0
while i < len(true_label):
    if true_label[i] == 1:
        count_zeros_before = 0
        j = i - 1
        while j >= 0 and true_label[j] == 0:
            count_zeros_before += 1
            j -= 1
        
        count_zeros_after = 0
        k = i + 1
        while k < len(true_label) and true_label[k] == 0:
            count_zeros_after += 1
            k += 1

        if count_zeros_before >= 2 and count_zeros_after >= 2:
            true_label[i] = 0
            i += 3
        else:
            i += 1
    else:
        i += 1
        
print(f"true label after processing \n {true_label}")   
print(f"predicted label after processing \n {all_predictions}")      

precision = precision_score(true_label, all_predictions)
recall = recall_score(true_label, all_predictions)
f1 = f1_score(true_label, all_predictions)

print(f'Precision after processing: {precision:.4f}')
print(f'Recall after processing: {recall:.4f}')
print(f'F1 Score after processing: {f1:.4f}')
