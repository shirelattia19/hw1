import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader, SubsetRandomSampler, Subset
import math


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        # TODO:
        # Implement the logic required for this sampler.
        # If the length of the data source is N, you should return indices in a
        # first-last ordering, i.e. [0, N-1, 1, N-2, ...].
        # ====== YOUR CODE: ======
        res = []
        data_length = len(self.data_source)
        for idx in range(math.ceil(data_length / 2)):
            res.append(idx)
            if idx < data_length - idx - 1:
                res.append(data_length - idx - 1)
        return iter(res)
        # ========================

    def __len__(self):
        return len(self.data_source)


def create_train_validation_loaders(
        dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    # TODO:
    #  Create two DataLoader instances, dl_train and dl_valid.
    #  They should together represent a train/validation split of the given
    #  dataset. Make sure that:
    #  1. Validation set size is validation_ratio * total number of samples.
    #  2. No sample is in both datasets. You can select samples at random
    #     from the dataset.
    #  Hint: you can specify a Sampler class for the `DataLoader` instance
    #  you create.
    # ====== YOUR CODE: ======
    # train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=validation_ratio)
    # ds_train = Subset(dataset, train_idx)
    # ds_valid = Subset(dataset, val_idx)
    #ds_train, ds_valid = train_test_split(dataset, test_size=validation_ratio)

    array = np.arange(len(dataset))
    np.random.shuffle(array)
    train_idx = array[0:int((1 - validation_ratio) * len(dataset))]
    valid_idx = array[int((1 - validation_ratio) * len(dataset)):]
    dl_train = DataLoader(dataset, batch_size=batch_size,
                          num_workers=num_workers, sampler=SubsetRandomSampler(train_idx))
    dl_valid = DataLoader(dataset, batch_size=batch_size,
                          num_workers=num_workers, sampler=SubsetRandomSampler(valid_idx))
    # ========================
    return dl_train, dl_valid
