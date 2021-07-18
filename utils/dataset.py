import pickle
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler
from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from pathlib import Path

#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)
#torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)

###################################################################################
# WaveRNN/Vocoder Dataset #########################################################
###################################################################################
def get_datasets(path, batch_size, shuffle):
    with open(f'{path}/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    ids = [x[0] for x in dataset]
    lens = [x[1] for x in dataset]
    
    lens, ids = zip(*sorted(zip(lens,ids),reverse=True))
    train_dataset = RTestDataset(path, ids)
    point_sampler=SequentialSampler(train_dataset)
    batch_sampler=SuffleBatchSampler(point_sampler,batch_size)

    train_set = DataLoader(train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=colla_RTest, 
            pin_memory=True, num_workers=1,
            worker_init_fn=seed_worker)
    return train_set 

def colla_RTest(samples):
    DR=[torch.from_numpy(sample[0].T) for sample in samples]
    T60=[torch.from_numpy(sample[1]) for sample in samples]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(DR,batch_first=True,padding_value=0.0)
    return padded_inputs.contiguous(), torch.stack(T60).contiguous()

class RTestDataset(Dataset):
    def __init__(self, path, dataset_ids):
        self.metadata = dataset_ids
        self.path = path
    
    def __getitem__(self, index):
        utt_id= self.metadata[index]
        DR = np.load(f'{self.path}/decayRates/{utt_id}.npy')
        T60 = np.load(f'{self.path}/T60/{utt_id}.npy')
        return DR, T60

    def __len__(self):
        return len(self.metadata)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SuffleBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Sampler[int], batch_size: int) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.sampler) // self.batch_size 
        generator=torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        if self.batch_size == 1:
            for idx in torch.randperm(n, generator=generator).tolist():
                if idx == n:
                    yield range(idx*self.batch_size,len(self.sampler))
                else:
                    yield range(idx*self.batch_size,(idx+1)*self.batch_size)

        else:
            for idx in torch.randperm(n+1, generator=generator).tolist():
                if idx == n:
                    yield range(idx*self.batch_size,len(self.sampler))
                else:
                    yield range(idx*self.batch_size,(idx+1)*self.batch_size)

    def __len__(self) -> int:
        return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
