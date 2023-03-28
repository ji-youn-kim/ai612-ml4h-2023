# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
)
from argparse import Namespace

from .lr_scheduler import LRScheduler
from utils import setup_registry

(
    build_lr_scheduler_,
    register_lr_scheduler,
    LR_SCHEDULER_REGISTRY
) = setup_registry(
    "--lr-scheduler", base_class=LRScheduler, default="fixed"
)

def build_lr_scheduler(args: Namespace, optimizer):
    return build_lr_scheduler_(args, optimizer)

# automatically import any Python files in the optim/lr_scheduler/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("optim.lr_scheduler." + file_name)