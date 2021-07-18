import numpy as np
import pickle
#import glob

from utils.display import *
#import scipy
#import librosa
#import librosa.display

from utils.dataset import get_datasets
import matplotlib.pyplot as plt
import argparse, os, json, random
from models.Model_structure import RT_est
from matplotlib import gridspec
from utils.paths import NetPaths
from utils.checkpoints import save_checkpoint, restore_checkpoint
import torch
from torch import optim
import torch.nn.functional as F
import seaborn as sns
from pandas import DataFrame

#for Visualization
gs=gridspec.GridSpec(nrows=1,ncols=1,hspace=0.5,wspace=0.1)
graphColor=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#FE parameters
nFbank=40
##
#Training parameters
fc_dim=512
def weights_init_normal(m):
    if isinstance(m,torch.nn.Conv1d): 
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)
    if isinstance(m,torch.nn.Linear): 
        torch.nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


def main():

    if torch.cuda.is_available():
        device = torch.device('cuda')
    print('Using device:', device)

    #Network 
    SavePaths = NetPaths('TrainSet1')
    TNmodel = RT_est(num_channels=nFbank, fc_dim=fc_dim).to(device)
    TNoptimizer = optim.Adam(TNmodel.parameters(), weight_decay=1e-3)

    print('\nInitializing RT_est Network\n')
    restore_checkpoint(SavePaths, TNmodel, TNoptimizer, create_if_missing=True)

    test_TN(SavePaths, TNmodel, TNoptimizer)

def test_TN(paths:NetPaths, model:RT_est, optimizer):
    #EvalSet
    evalDir='../FE_dir_3_shortframe/data-ACEeval'
    with open(f'{evalDir}/dataset.pkl','rb') as f:
        dataset = pickle.load(f)
    evalset_ids = [x[0] for x in dataset]

    #train setting
    device = next(model.parameters()).device
    gtT60=[0.6570, 0.6950, 1.3801, 1.3485, 0.4671, 0.4648, 0.4148, 0.4071, 0.4305, 0.4217]
    RIRs=['Lecture_Room_1_1','Lecture_Room_1_2','Lecture_Room_2_1','Lecture_Room_2_2','Meeting_Room_1_1','Meeting_Room_1_2','Meeting_Room_2_1','Meeting_Room_2_2','Office_2_1','Office_2_2']
    Rooms=['Lecture_Room_1','Lecture_Room_2','Meeting_Room_1','Meeting_Room_2','Office_2']
    df = DataFrame(columns=['Sentence','Room','RIR','Speaker','Gender','NoiseType','SNR Level','RT_gt', 'RT_est', 'err(s)'])
    for i, utt_id in enumerate(evalset_ids):
        DR = np.load(f'{evalDir}/decayRates/{utt_id}.npy')[-nFbank:,:]; DR=torch.from_numpy(DR.T); DR=DR.view(1,-1,nFbank)
        DR = DR.to(device,torch.float32)
        est_T60, *_ = model(DR)
        est_T60 = est_T60.to('cpu').detach().numpy()
        if utt_id.split('_')[1] == 'Office':
            if utt_id.split('_')[3] == '1':
                RIR = RIRs[8]; Room = Rooms[4]; RT_gt = gtT60[8]
            else:
                RIR = RIRs[9]; Room = Rooms[4]; RT_gt = gtT60[9]
        elif utt_id.split('_')[1] == 'Meeting':
            if utt_id.split('_')[3] == '1':
                if utt_id.split('_')[4] == '1':
                    RIR = RIRs[4]; Room = Rooms[2]; RT_gt = gtT60[4]
                else:
                    RIR = RIRs[5]; Room = Rooms[2]; RT_gt = gtT60[5]
            else:
                if utt_id.split('_')[4] == '1':
                    RIR = RIRs[6]; Room = Rooms[3]; RT_gt = gtT60[6]
                else:
                    RIR = RIRs[7]; Room = Rooms[3]; RT_gt = gtT60[7]
        else:
            if utt_id.split('_')[3] == '1':
                if utt_id.split('_')[4] == '1':
                    RIR = RIRs[0]; Room = Rooms[0]; RT_gt = gtT60[0]
                else:
                    RIR = RIRs[1]; Room = Rooms[0]; RT_gt = gtT60[1]
            else:
                if utt_id.split('_')[4] == '1':
                    RIR = RIRs[2]; Room = Rooms[1]; RT_gt = gtT60[2]
                else:
                    RIR = RIRs[3]; Room = Rooms[1]; RT_gt = gtT60[3]
        SNRLevel=utt_id.split('_')[-1]; NoiseType=utt_id.split('_')[-2]; SentenceType=utt_id.split('_')[-3]; Speaker=utt_id.split('_')[-4]; 
        if Speaker[0] == 'F':
            Gender = 'Female'
        else:
            Gender = 'Male'
        bar = progbar(i+1, len(evalset_ids))
        message = f'{bar} {i+1}/{len(evalset_ids)} '
        stream(message)
        df = df.append(DataFrame([[SentenceType,Room,RIR,Speaker,Gender,NoiseType,SNRLevel,RT_gt,est_T60,RT_gt-est_T60]],columns=['Sentence','Room','RIR','Speaker','Gender','NoiseType','SNR Level','RT_gt', 'RT_est', 'err(s)']),ignore_index=True)
    df['RT_est'] = df['RT_est'].astype(float)
    df['RT_gt'] = df['RT_gt'].astype(float)
    df['err(s)'] = df['err(s)'].astype(float)
    print('\n')
    print('Bias:', df['err(s)'].mean(), 'MSE:', np.square(df['err(s)']).mean())
    print(df.corr(method='pearson'))
    print('s1(MSE):',np.square(df.groupby(['Sentence']).get_group('s1')['err(s)']).mean())
    print('s2(MSE):',np.square(df.groupby(['Sentence']).get_group('s2')['err(s)']).mean())
    print('s3(MSE):',np.square(df.groupby(['Sentence']).get_group('s3')['err(s)']).mean())
    print('s4(MSE):',np.square(df.groupby(['Sentence']).get_group('s4')['err(s)']).mean())
    print('s5(MSE):',np.square(df.groupby(['Sentence']).get_group('s5')['err(s)']).mean())

    print('Lecture_Room_1(MSE):',np.square(df.groupby(['Room']).get_group('Lecture_Room_1')['err(s)']).mean())
    print('Lecture_Room_2(MSE):',np.square(df.groupby(['Room']).get_group('Lecture_Room_2')['err(s)']).mean())
    print('Meeting_Room_1(MSE):',np.square(df.groupby(['Room']).get_group('Meeting_Room_1')['err(s)']).mean())
    print('Meeting_Room_2(MSE):',np.square(df.groupby(['Room']).get_group('Meeting_Room_2')['err(s)']).mean())
    print('Office_2(MSE):',np.square(df.groupby(['Room']).get_group('Office_2')['err(s)']).mean())

    fig=plt.figure(1,figsize=(12,6))
    sns.boxplot(x='Room', y='err(s)', data=df, fliersize=0.1,order=['Meeting_Room_2','Office_2','Meeting_Room_1','Lecture_Room_1','Lecture_Room_2'])
    plt.ylim([-1.0, 1.0])
    fig.savefig('/Database/DB1/RT_err_1.png')

if __name__ == "__main__":
    main()
