# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from .distributed_model import DistributedModel
from .model import BaseModel

__all__ = [
    "BaseModel",
    "DistributedModel"
]

# registry
MODEL_REGISTRY = {}

def build_model(name, **kwargs):
    model = None

    if name not in MODEL_REGISTRY:
        raise Exception(
            "Could not infer model class from directory. Please register model by decorating "
            "the indicated class with @register_model('...'). "
            "Available models: "
            + str(MODEL_REGISTRY.values())
            + " Requested model: "
            + name
        )
    
    model = MODEL_REGISTRY[name]
    
    return model.build_model(**kwargs)

def register_model(name):
    """
    New model types can be added to this framework with the :func:`register_model`
    function decorator.
    
    Usage:
        @register_model("lstm")
        class LSTM(BaseModel):
            (...)
    
    Args:
        name (str): the name of the model
    
    Notes:
        All models must implement the :class:`BaseModel` interface.
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        
        MODEL_REGISTRY[name] = cls
        
        return cls

    return register_model_cls

def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            model_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + model_name)

# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "models")