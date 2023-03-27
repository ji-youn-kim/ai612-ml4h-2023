# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

class BaseDataset(torch.utils.data.Dataset):
    """A dataset that provides helpers for batching"""
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.
        
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        raise NotImplementedError
    
    @classmethod
    def build_dataset(cls, **kwargs):
        """Build a new dataset instance."""
        return cls(**kwargs)