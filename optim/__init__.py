# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import importlib

from utils import setup_registry
from .optimizer import Optimizer

from argparse import Namespace

__all__ = [
    "Optimizer"
]

(
    _build_optimizer,
    register_optimizer,
    OPTIMIZER_REGISTRY
) = setup_registry("--optimizer", base_class=Optimizer, required=True)

def build_optimizer(cfg: Namespace, params, *extra_args, **extra_kwargs):
    if all(isinstance(p, dict) for p in params):
        params = [t for p in params for t in p.values()]
    return _build_optimizer(cfg, params, *extra_args, **extra_kwargs)

# automatically import any Python files in the optim/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("optim." + file_name)