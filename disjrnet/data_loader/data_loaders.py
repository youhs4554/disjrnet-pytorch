import os
from pathlib import Path
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from base import BaseDataLoader
from .dataset import VideoDataset
import random


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class VideoDataLoader(BaseDataLoader):
    """
        General data loader for fall three different video datasets: FDD, URFD, MulticamFD
    """

    def __init__(self, root, batch_size, shuffle=False, validation_split=0.25, num_workers=1, fold=1,
                 sample_length=10,
                 sample_step=1,
                 sample_stride=1,
                 video_ext=".avi",
                 use_person_label=False, training=True):
        if training:
            split_file = "trainlist{:02}.txt".format(fold)
            _collate_fn = self.balanced_random_collate_fn
        else:
            split_file = "testlist{:02}.txt".format(fold)
            _collate_fn = default_collate

        self.dataset = VideoDataset(
            root=root, split_file=os.path.join(
                root, "TrainTestlist", split_file
            ),
            sample_length=sample_length,
            sample_step=sample_step,
            sample_stride=sample_stride,
            video_ext=video_ext,
            use_person_label=use_person_label)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, collate_fn=_collate_fn)

        self.fold = fold

    def balanced_random_collate_fn(self, samples):
        random.shuffle(samples)
        return default_collate(samples)
