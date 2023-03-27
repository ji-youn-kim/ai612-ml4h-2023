# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class BaseModel(nn.Module):
    """Base class for various models"""

    def __init__(self):
        super().__init__()
        
        self.num_updates = 0
    
    @classmethod
    def build_model(cls, **kwargs):
        """Build a new model instance."""
        return cls(**kwargs)
    
    def get_logits(cls, net_output):
        """get logits from the net's output."""
        raise NotImplementedError("Model must implement the get_logits method")
    
    def get_targets(self, sample):
        """get targets from the sample"""
        raise NotImplementedError("Model must implement the get_targets method")
    
    def set_num_updates(self, num_updates):
        """State from traininer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)