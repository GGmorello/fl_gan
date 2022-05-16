# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mnist Shard Descriptor."""

import logging
import os
from typing import List

import numpy as np
import requests

from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from keras.datasets.mnist import load_data
from numpy import expand_dims

logger = logging.getLogger(__name__)


class MnistShardDataset(ShardDataset):
    """Mnist Shard dataset class."""

    def __init__(self, x, data_type, rank=1, worldsize=1):
        """Initialize TinyImageNetDataset."""
        self.data_type = data_type
        self.rank = rank
        self.worldsize = worldsize
        self.x = x[self.rank - 1::self.worldsize]

    def __getitem__(self, index: int):
        """Return an item by the index."""
        return self.x[index]

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.x)


class MnistShardDescriptor(ShardDescriptor):
    """Mnist Shard descriptor class."""

    def __init__(
            self,
            rank_worldsize: str = '1, 1',
            **kwargs
    ):
        """Initialize MnistShardDescriptor."""
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))
        x_train = self.download_data()
        self.data_by_type = {
            'data': x_train,
        }

    def get_shard_dataset_types(self) -> List[str]:
        """Get available shard dataset types."""
        return list(self.data_by_type)

    def get_dataset(self, dataset_type):
        """Return a shard dataset by type."""
        if dataset_type not in self.data_by_type:
            raise Exception(f'Wrong dataset type: {dataset_type}')
        return MnistShardDataset(
            self.data_by_type[dataset_type],
            data_type='data',
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        """Return the sample shape info."""
        return ['784']

    @property
    def target_shape(self):
        """Return the target shape info."""
        return ['1']

    @property
    def dataset_description(self) -> str:
        """Return the dataset description."""
        return (f'Mnist dataset, shard number {self.rank}'
                f' out of {self.worldsize}')

    def download_data(self):
        """Download prepared dataset."""
        (x_train, y_train), (x_test, y_test) = load_data()
        X = expand_dims(x_train, axis=-1)
        X = X.astype('float32')
        X = X / 255.0
        return X
