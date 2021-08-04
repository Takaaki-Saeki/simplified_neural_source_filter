import json
import math
import os
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
import torch

class Dataset(Dataset):
    
    def __init__(self, filetxt, config, device):
        
        self.preprocessed_path = config["path"]["preprocessed_path"]
        with open(os.path.join(self.preprocessed_path, filetxt), 'r') as fr:
            self.filelists = [path.strip('\n') for path in fr]
        self.device = device

        self.f0s = []
        self.sps = []
        self.wavs = []
        for filepath in self.filelists:

            if config["preprocess"]["corpus"] == "jvs":
                basename = os.path.basename(os.path.dirname(filepath)) + "-" + os.path.splitext(os.path.basename(filepath))[0]
            elif config["preprocess"]["corpus"] == "jsut":
                basename = os.path.splitext(os.path.basename(filepath))[0]

            f0 = np.load(os.path.join(self.preprocessed_path, "f0", "f0-{}.npy".format(basename)))
            self.f0s.extend(f0)
            sp = np.load(os.path.join(self.preprocessed_path, "sp", "sp-{}.npy".format(basename)))
            self.sps.extend(sp)
            wav = np.load(os.path.join(self.preprocessed_path, "wav", "wav-{}.npy".format(basename)))
            self.wavs.extend(wav)

        self.f0s = np.asarray(self.f0s)
        self.sps = np.asarray(self.sps)
        self.wavs = np.asarray(self.wavs)

    def __len__(self):
        return self.f0s.shape[0]

    def __getitem__(self, idx):

        f0 = self.f0s[idx, :]
        sp = self.sps[idx, :, :]
        wav = self.wavs[idx, :]

        sample = {
            'f0': f0,
            'sp': sp,
            'wav': wav
        }

        return sample

    def collate_fn(self, batch):

        batch_size = len(batch)

        f0s = []
        sps = []
        wavs = []

        for idx in range(batch_size):
            f0s.append(batch[idx]['f0'])
            sps.append(batch[idx]['sp'])
            wavs.append(batch[idx]['wav'])
        f0s = np.asarray(f0s)
        sps = np.asarray(sps)
        wavs = np.asarray(wavs)
        
        f0s = torch.from_numpy(f0s).to(self.device)
        sps = torch.from_numpy(sps).to(self.device)
        wavs = torch.from_numpy(wavs).to(self.device)
        feature = torch.cat((f0s.unsqueeze(1), sps), dim=1)

        output = [
            f0s,
            sps,
            feature,
            wavs
        ]

        return output

if __name__ == '__main__':

    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    train_dataset = Dataset('train.txt', 'preprocessed', device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )