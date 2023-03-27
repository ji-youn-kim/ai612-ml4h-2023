import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import importlib

from utils import setup_registry

from .dataset import BaseDataset

from argparse import Namespace

(
    build_dataset_,
    register_dataset,
    DATASET_REGISTRY
) = setup_registry("--dataset", base_class=BaseDataset, required=True)

def build_dataset(args: Namespace):
    return build_dataset_(**vars(args))

# automatically import any Python files in the criterions/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("data." + file_name)