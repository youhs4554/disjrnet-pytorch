import copy
import random
import torch
from torch.utils.data.distributed import DistributedSampler


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    Link : https://github.com/pytorch/pytorch/issues/23430#issuecomment-562350407

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(
            sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(
                len(indices), self.total_size))

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(
                len(indices), self.num_samples))

        return iter(indices)


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)

        self.indices = [-1]*len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return int(self.labels[idx])
        else:
            # raise Exception(
            #     "You should pass the tensor of labels to the constructor as second argument")
            return dataset[idx]["label"]

    def __len__(self):
        return self.balanced_max*len(self.keys)


def test_sampler():
    import torch
    epochs = 3
    size = 465
    features = 5
    classes_prob = torch.tensor([0.1, 0.9])

    dataset_X = torch.randn(size, features)
    dataset_Y = torch.distributions.categorical.Categorical(
        classes_prob.repeat(size, 1)).sample()

    dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)
    dataset = copy.deepcopy(
        torch.utils.data.dataset.Subset(dataset, range(300)))

    train_loader = torch.utils.data.DataLoader(
        dataset, sampler=BalancedBatchSampler(dataset, dataset_Y), batch_size=8, num_workers=8)

    import ipdb
    ipdb.set_trace()

    for epoch in range(0, epochs):
        for batch_x, batch_y in train_loader:
            print("epoch: %d labels: %s\n" %
                  (epoch, torch.unique(batch_y, return_counts=True)))
