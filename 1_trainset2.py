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


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


#for Visualization
gs=gridspec.GridSpec(nrows=1,ncols=1,hspace=0.5,wspace=0.1)
graphColor=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#FE parameters
nFbank=40

##
#trainDir='../FE_dir_3_multiscale/data-train'
#evalDir='../FE_dir_3_multiscale/data-ACEeval'
#devDir='../FE_dir_3_multiscale/data-ACEdev'
trainDir='../FE_dir_3_shortframe/data-trainset2'
evalDir='../FE_dir_3_shortframe/data-ACEeval'
devDir='../FE_dir_3_shortframe/data-ACEdev'

#Training parameters
batch_size=16
lr=1e-3
fc_dim=512
teacher_steps=200_000
max_epochs=100
decay_epoch=50
decay_lr=0.95
eval_check=2
update_period=16

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
    SavePaths = NetPaths('TrainSet2')
    TNmodel = RT_est(num_channels=nFbank, fc_dim=fc_dim).to(device)
    TNoptimizer = optim.Adam(TNmodel.parameters(), lr=lr, weight_decay=1e-3)
    TNmodel.apply(weights_init_normal)


    print('\nInitializing RT_est Network\n')
    restore_checkpoint(SavePaths, TNmodel, TNoptimizer, create_if_missing=True)

    train_set = get_datasets(trainDir,batch_size,False)

    train_TN(SavePaths, TNmodel, TNoptimizer, train_set, teacher_steps)

def train_TN(paths:NetPaths, model:RT_est, optimizer, train_set, teacher_steps):
    #EvalSet
    with open(f'{evalDir}/dataset.pkl','rb') as f:
        dataset = pickle.load(f)
    evalset_ids = [x[0] for x in dataset]
    #DevSet
    with open(f'{devDir}/dataset.pkl','rb') as f:
        dataset = pickle.load(f)
    devset_ids = [x[0] for x in dataset]

    #train setting
    device = next(model.parameters()).device
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_lr)
    total_iters = len(train_set)
    epochs = max_epochs 

#    Tensor = torch.cuda.FloatTensor
    MSEloss_function = torch.nn.MSELoss()

    #Loss_check
    trainLoss=[]
    trainpoint=[]
    evalLoss=[]
    evalpoint=[]
    devLoss=[]
    s1Loss=[]
    s2Loss=[]
    s3Loss=[]
    s4Loss=[]
    s5Loss=[]

    for e in range(1, epochs +1):
        start = time.time()
        running_loss = 0.
        model.train()
        optimizer.zero_grad()
        num_period=0
        for i, (DR, T60) in enumerate(train_set, 1):
            DR, T60 = DR[:,:,-nFbank:].to(device,torch.float32), T60.to(device,torch.float32)
            est_T60, *_ = model(DR)
            loss = MSEloss_function(est_T60, T60.view(-1,1))
            loss.backward()
            num_period += 1
            if num_period == update_period:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                if np.isnan(grad_norm.cpu()):
                    print('grad_norm was NaN!')
                optimizer.step()
                optimizer.zero_grad()
                num_period=0
            running_loss += loss.item()
            avg_loss = running_loss / i
            speed = i / (time.time() - start)
            model.step_update()
            step = model.get_step()
            k = step // 1_000

            for g in optimizer.param_groups: updateLr = g['lr']
            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | Learning rate: {updateLr:.6f} | {speed:.1f} steps/s | Step: {k}k '
            stream(msg)
        trainLoss.append(avg_loss)
        trainpoint.append(e)
        save_checkpoint(paths, model, optimizer, is_silent=True)
        if e>decay_epoch:
            lr_scheduler.step()
        model.eval()
        if e % eval_check == 0:
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
            print('Bias:', np.abs(df['err(s)']).mean(), 'MSE:', np.square(df['err(s)']).mean())
            evalLoss.append(np.square(df['err(s)']).mean())
#            if min(evalLoss) == df['MSE'].mean():
#                print('save Model')
#                save_checkpoint(paths, model, optimizer, is_silent=True)
            s1Loss.append(np.square(df.groupby(['Sentence']).get_group('s1')['err(s)']).mean())
            s2Loss.append(np.square(df.groupby(['Sentence']).get_group('s2')['err(s)']).mean())
            s3Loss.append(np.square(df.groupby(['Sentence']).get_group('s3')['err(s)']).mean())
            s4Loss.append(np.square(df.groupby(['Sentence']).get_group('s4')['err(s)']).mean())
            s5Loss.append(np.square(df.groupby(['Sentence']).get_group('s5')['err(s)']).mean())
            evalpoint.append(e)

            gtT60=[0.8667,0.8218,0.3811,0.3244]
            RIRs=['Building_Lobby_1','Building_Lobby_2','Office_1_1','Office_1_2']
            Rooms=['Building_Lobby','Office_1']
            df = DataFrame(columns=['Room','RIR','Speaker','Gender','NoiseType','SNR Level','RT_gt', 'RT_est', 'err(s)'])
            for i, utt_id in enumerate(devset_ids):
                DR = np.load(f'{devDir}/decayRates/{utt_id}.npy')[-nFbank:,:]; DR=torch.from_numpy(DR.T); DR=DR.view(1,-1,nFbank)
                DR = DR.to(device,torch.float32)
                est_T60, *_ = model(DR)
                est_T60 = est_T60.to('cpu').detach().numpy()
                if utt_id.split('_')[1] == 'Office':
                    if utt_id.split('_')[3] == '1':
                        RIR = RIRs[2]; Room = Rooms[1]; RT_gt = gtT60[2]
                    else:
                        RIR = RIRs[3]; Room = Rooms[1]; RT_gt = gtT60[3]
                else:
                    if utt_id.split('_')[3] == '1':
                        RIR = RIRs[0]; Room = Rooms[0]; RT_gt = gtT60[0]
                    else:
                        RIR = RIRs[1]; Room = Rooms[0]; RT_gt = gtT60[1]
                SNRLevel=utt_id.split('_')[-1]; NoiseType=utt_id.split('_')[-2]; Speaker=utt_id.split('_')[-4]
                if Speaker[0] == 'F':
                    Gender = 'Female'
                else:
                    Gender = 'Male'
                bar = progbar(i+1, len(devset_ids))
                message = f'{bar} {i+1}/{len(devset_ids)} '
                stream(message)
                df = df.append(DataFrame([[Room,RIR,Speaker,Gender,NoiseType,SNRLevel,RT_gt,est_T60,RT_gt-est_T60]],columns=['Room','RIR','Speaker','Gender','NoiseType','SNR Level','RT_gt', 'RT_est', 'err(s)']),ignore_index=True)
            df['RT_est'] = df['RT_est'].astype(float)
            df['RT_gt'] = df['RT_gt'].astype(float)
            df['err(s)'] = df['err(s)'].astype(float)
            print('Bias:', np.abs(df['err(s)']).mean(), 'MSE:', np.square(df['err(s)']).mean())
            devLoss.append(np.square(df['err(s)']).mean())

            fig=plt.figure(1,figsize=(7,6))
            ax=fig.add_subplot(gs[0])
            ax.plot(trainpoint,trainLoss,color=graphColor[0])
            ax.plot(evalpoint,evalLoss,color=graphColor[1])
            ax.plot(evalpoint,devLoss,color=graphColor[2])
            ax.plot(evalpoint,s1Loss,color=graphColor[3])
            ax.plot(evalpoint,s2Loss,color=graphColor[4])
            ax.plot(evalpoint,s3Loss,color=graphColor[5])
            ax.plot(evalpoint,s4Loss,color=graphColor[6])
            ax.plot(evalpoint,s5Loss,color=graphColor[7])
            ax.legend(['Train','Eval','Dev','s1','s2','s3','s4','s5'])
            ax.set_ylim(0,0.1)
            ax.set_yticks(np.arange(0,0.1,step=0.01))
            fig.savefig('/Database/DB1/Loss_tracking.png')

    print('EvalLoss:',evalLoss)
    print('DevLoss:',devLoss)
    print('s1Loss:',s1Loss)
    print('s2Loss:',s2Loss)
    print('s3Loss:',s3Loss)
    print('s4Loss:',s4Loss)
    print('s5Loss:',s5Loss)



    

if __name__ == "__main__":

    main()
