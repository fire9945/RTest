import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
from pathlib import Path
from typing import Union

class RT_est(nn.Module):
    def __init__(self, num_channels, fc_dim):
        super().__init__()
        self.num_channels=num_channels

        self.est_weight=nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels*4, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*4, out_channels=num_channels*8, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*8, out_channels=num_channels*4, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*4, out_channels=num_channels*2, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=11, stride=1, padding=5),
            nn.Softmax(dim=2))

        self.estRT=nn.Sequential(
            nn.Linear(num_channels,fc_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_dim,fc_dim),
            nn.LeakyReLU(),
            nn.Linear(fc_dim,1)
            )

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.num_params()

    def forward(self, DR):
 #       temp = self.layer1(DR.transpose(1,2))
 #       temp = self.layer2(torch.add(temp,DR.transpose(1,2)))
 #       weight= self.est_weight(torch.add(DR.transpose(1,2),temp)).transpose(1,2)
        weight= self.est_weight(DR.transpose(1,2)).transpose(1,2)
        
        stat = torch.mul(weight,DR)
        stat = torch.sum(stat,dim=1)
        T60 = self.estRT(stat)

        return T60, weight, stat

    def step_update(self):
        self.step += 1

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg,file=f)

    def load(self, path: Union[str, Path]):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]

class RT_est_SA(nn.Module):
    def __init__(self, num_channels, fc_dim):
        super().__init__()
        self.num_channels=num_channels

        self.est_weight=nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels*4, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*4, out_channels=num_channels*8, kernel_size=11, stride=1, padding=5, groups=num_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*8, out_channels=num_channels*4, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*4, out_channels=num_channels*2, kernel_size=11, stride=1, padding=5),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=11, stride=1, padding=5),
            nn.Softmax(dim=2))

        self.estRT_1=nn.Sequential(
                nn.Linear(num_channels,fc_dim),
                nn.LeakyReLU(),
                nn.Linear(fc_dim,fc_dim),
                nn.LeakyReLU(),
                nn.Linear(fc_dim,1))

        self.estRT_2=nn.Sequential(
                nn.Linear(num_channels,fc_dim),
                nn.LeakyReLU(),
                nn.Linear(fc_dim,fc_dim),
                nn.LeakyReLU(),
                nn.Linear(fc_dim,1))

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.num_params()

    def forward(self, DR, SegmentSize):
        SegmentDR = []
        DRsize = DR.shape[1]

        overlap = int(SegmentSize / 10)
        if DRsize > SegmentSize:
            step = SegmentSize - overlap
            iter = int((DRsize - overlap) / step) + 1
            for i in range(iter):
                if i == 0:
                    SegmentDR.append(DR[:,:SegmentSize,:])
                elif i < iter -1:
                    SegmentDR.append(DR[:,i*step:i*step + SegmentSize,:])

                else:
                    SegmentDR.append(DR[:,-SegmentSize:,:])
        else:
            iter = 1
            SegmentDR.append(DR)
        for i in range(iter):
            DR = SegmentDR[i]

            weight = self.est_weight(DR.transpose(1,2)).transpose(1,2)

            stat = torch.mul(weight,DR)
            stat = torch.sum(stat,dim=1)

            if i == 0: stat_SA = stat.unsqueeze(1)
            else: stat_SA = torch.cat([stat_SA, stat.unsqueeze(1)], dim = 1)

            T60 = self.estRT_2(stat)
            if i == 0: T60_SA = T60

            else: T60_SA = torch.cat([T60_SA, T60], dim = 1)

        T60 = self.estRT_1(torch.mean(stat_SA,dim = 1))

        return T60, T60_SA, weight, torch.mean(stat_SA,dim = 1)

    def step_update(self):
        self.step += 1

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg,file=f)

    def load(self, path: Union[str, Path]):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters

    def _flatten_parameters(self):
        [m.flatten_parameters() for m in self._to_flatten]

