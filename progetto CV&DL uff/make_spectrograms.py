import os, sys
import numpy as np
import torch
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
torch.manual_seed(184)
torch.cuda.manual_seed(184)


# SET SOME PARAMETERS

wav_path = "/disks/disk1/adanna/PSG_Audio/EDF/"
dest_path = "/disks/disk1/adanna/MELSP_6S/"
meta_file = "/home/adanna/Codice/apnea_detection_v3/meta.csv"
SR = 16000 # working sample rate

patients = [ 1014, 1106, 1037,1008,1120, 1045, 1016, 1059, 1041, 
       1028, 1095, 1084, 1093, 1112, 1073, 1104, 1043, 1018,
        1082, 1022, 1110, 1039, 1089,  1116, 1010, 1069, 1118, 1086, 1024, 1071] # Patient number without leading zeros

E_type = ['ObstructiveApnea', 'CentralApnea', 'MixedApnea', 'Hypopnea'] # desired apnea type(s)
context_s = 6 # secondi di contesto


for P_n in patients:
    print(f'Patient number {P_n}')
    # create destination folders or rise error if already exist
    patient_dest_path = os.path.join(dest_path, 'P'+str(P_n))
    apnea_dest_path = os.path.join(dest_path, 'P'+str(P_n), 'apnea')
    nonapnea_dest_path = os.path.join(dest_path, 'P'+str(P_n), 'nonapnea')
    if not os.path.exists(patient_dest_path):
        os.makedirs(patient_dest_path)
        os.makedirs(apnea_dest_path)
        os.makedirs(nonapnea_dest_path)
    else:
        raise Exception(f'Destination folder for patient {str(P_n)} already exists!')

    meta_df = pd.read_csv(meta_file, low_memory=False)
    apnea_df = meta_df.loc[(meta_df['Type'] == E_type[0])|(meta_df['Type'] == E_type[1])|(meta_df['Type'] == E_type[2])|(meta_df['Type'] == E_type[3])]
    apnea_df = apnea_df.loc[apnea_df["Patient_ID"] == P_n]


    wav_files = os.listdir(os.path.join(wav_path, ("0000" + str(P_n))))
    s_len = 0.0
    for wf in wav_files:
        # get wav file duration without opening
        d = librosa.get_duration(path = os.path.join(wav_path, ("0000" + str(P_n)), wf))
        s_len += d

    #BUILD LABEL MASK
    apneas_mask = torch.zeros(int(s_len*SR))
    for idx, row in apnea_df.iterrows():
        start = int(row['Start']*SR)
        dur = int(row['Duration']*SR)
        apneas_mask[start:start + dur] = 1.0

    #concatenate 1-hour audio files into a single array
    y_block = np.zeros((len(wav_files), 3600*SR))
    for w in range(len(wav_files)):
        fname = "0000" + str(P_n) + "-100507[00" + str(w+1) + "].wav"
        y, _ = librosa.load(os.path.join(wav_path, ("0000" + str(P_n)), fname), sr=SR)
        y_block[w,0:y.shape[0]] += y
    y_all = y_block.reshape(-1)


    #save MEL spectrograms into two separate folders: apnea and nonapnea
    for n, c in enumerate(tqdm(range(0,apneas_mask.shape[0],(context_s*SR)))):
        if sum(apneas_mask[c:c+(context_s*SR)]) == 0: # IF NON APNEA
            try:
                y_chunk = y_all[c:c+(context_s*SR)]
                S2 = librosa.feature.melspectrogram(y=y_chunk, sr=SR, hop_length=185, n_mels=64)
                S2_dB = librosa.power_to_db(S2, ref=np.max) # shape [64,519]
                S2_dB = torch.from_numpy(S2_dB).float()
                torch.save(S2_dB, os.path.join(nonapnea_dest_path, str(n) + '.pth'))
            except:
                print(f'error')
        elif sum(apneas_mask[c:c+(context_s*SR)]) == (context_s*SR): # IF FULL 6s APNEA
            try:
                y_chunk = y_all[c:c+(context_s*SR)]
                S2 = librosa.feature.melspectrogram(y=y_chunk, sr=SR, hop_length=185, n_mels=64)
                S2_dB = librosa.power_to_db(S2, ref=np.max) # shape [64,519]
                S2_dB = torch.from_numpy(S2_dB).float()
                torch.save(S2_dB, os.path.join(apnea_dest_path, str(n) + '.pth'))
            except:
                print(f'error')
