import numpy as np
import copy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from utils.sampler import BalancedBatchSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)
        self.dataset = dataset

        self.sampler, self.valid_sampler = self._split_sampler(
            self.validation_split)

        if hasattr(self, 'train_ds'):
            dataset = self.train_ds

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        labels_full = self.dataset.labels

        train_idx, valid_idx, train_labels, valid_labels = train_test_split(
            idx_full, labels_full, test_size=split, stratify=labels_full
        )

        train_ds = copy.deepcopy(Subset(self.dataset, train_idx))

        # turn off training mode and subsampling for validation
        tmp = copy.deepcopy(self.dataset)
        tmp.subsample_validationset(valid_idx.tolist())
        valid_ds = tmp
        valid_idx = range(len(valid_ds.dataset.indices))

        # for balanced batch
        train_sampler = BalancedBatchSampler(
            train_ds, labels=torch.tensor(train_labels))
        # valid_sampler = SubsetRandomSampler(valid_idx)
        valid_sampler = SequentialSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        # register validation dataset
        self.train_ds = train_ds
        self.valid_ds = valid_ds

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            # replace dataset with validation dataset
            self.init_kwargs["dataset"] = self.valid_ds
            self.init_kwargs["collate_fn"] = default_collate
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
