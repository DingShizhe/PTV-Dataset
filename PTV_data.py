from functools import partial
import os
import torch
import scipy.io as sio
import numpy as np

import pytorch_lightning as pl
import pdb


class PTVDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot):
        super(PTVDataset, self).__init__()
        self.dataroot = dataroot
        fns = sorted(list(os.listdir(self.dataroot)))
        fns = [ os.path.join(self.dataroot, fn) for fn in fns ]
        self.data = [np.load(fn) for fn in fns]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def _collate(data, test=True):

    for d in data:
        seed = int(d[0][0][0] * 1000)
        np.random.seed(seed)
        np.random.shuffle(d)

    # \times 2 because 20 frames / s
    data = [np.concatenate((d[:, 0, :], (d[:, 1, :] - d[:, 0, :]) * 2 ), axis=1) for d in data]

    for d in data:
        d[:,0] = (d[:,0] - 250) / 350       # normalize 
        d[:,1] = (d[:,1] - 350) / 350       # normalize 
        d[:,2] = d[:,2] / 20                # normalize
        d[:,3] = d[:,3] / 20                # normalize

    o_data = [d[:512] for d in data]
    t_data = [d[512:] for d in data]

    o_data = torch.from_numpy(np.array(o_data)).to(torch.float32)

    max_l = max([len(td) for td in t_data])
    t_mask = [np.ones(len(td)) for td in t_data]
    t_mask = [np.pad(tm, ((0, max_l-len(tm)))) for tm in t_mask]
    t_data = [np.pad(td, ((0, max_l-len(td)), (0,0))) for td in t_data]
    t_data = torch.from_numpy(np.array(t_data)).to(torch.float32)

    t_mask = torch.from_numpy(np.array(t_mask)).to(torch.float32)

    return o_data[:, :, :2], o_data[:, :, 2:]*2, t_data[:, :, :2], t_data[:, :, 2:]*2, 512, t_mask


class PTVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        train_path,
        test_path,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.data_root = data_root
        self.train_path = train_path
        self.test_path = test_path

        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        dataset = PTVDataset(self.data_root)

        # train test split
        train_idx = list(range(0, len(dataset), 4)) + list(range(1, len(dataset), 4)) + list(range(2, len(dataset), 4))
        test_idx = list(range(3, len(dataset), 4))

        self.train_val_dataset = torch.utils.data.Subset(dataset, train_idx)

        train_length, val_length = int(len(self.train_val_dataset) * 0.8), len(
                self.train_val_dataset
            ) - int(len(self.train_val_dataset) * 0.8)

        train_dataset, val_dataset = torch.utils.data.random_split(
                self.train_val_dataset, [train_length, val_length]
            )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = torch.utils.data.Subset(dataset, test_idx)

    def setup(self, stage=None):
        self.prepare_data()

    def __dataloader(self, dataset, shuffle=False, test=False):
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            collate_fn=partial(_collate, test=test),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return loader

    def train_dataloader(self):
        return self.__dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False, test=True)

    def test_dataloader(self):
        return self.__dataloader(self.test_dataset, shuffle=False, test=True)
