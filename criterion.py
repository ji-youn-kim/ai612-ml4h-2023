# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import inspect
from argparse import Namespace

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

import utils
from loggings import meters, metrics

class MultiTaskCriterion(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    @classmethod
    def build_criterion(cls, args: Namespace):
        """Construct a criterion from command-line args."""
        # arguments in the __init__.
        init_args = {}
        for p in inspect.signature(cls).parameters.values():
            if (
                p.kind == p.POSITIONAL_ONLY
                or p.kind == p.VAR_POSITIONAL
                or p.kind == p.VAR_KEYWORD
            ):
                raise NotImplementedError("{} not supported".format(p.kind))

            assert p.kind in {p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY}

            if p.name == "args":
                init_args["args"] = args
            elif hasattr(args, p.name):
                init_args[p.name] = getattr(args, p.name)
            elif p.default != p.empty:
                pass
            else:
                raise NotImplementedError(
                    "Unable to infer Criterion arguments, please implement "
                    "{}.build_criterion".format(cls.__name__)
                )
        return cls(**init_args)
    
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        
        Returns a tuple with three elements.
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        
        logits = model.get_logits(net_output).float() # (batch, 29, dim)
        target = model.get_targets(sample) # (batch, 29,)

        assert logits.dim() == 3 and target.dim() == 2

        reduction = "none" if not reduce else "sum"

        target_idx = torch.where(target != -1)
        logits = logits[target_idx]
        target = target[target_idx]
        probs = torch.sigmoid(logits)
        
        loss = F.binary_cross_entropy(
            inputs=probs,
            target=target,
            reduction=reduction
        )

        sample_size = len(target)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "batch_size": len(sample),
            "sample_size": sample_size
        }

        with torch.no_grad():
            y_class = [x.item() for x in target_idx[1]]
            y_true = target.cpu().numpy()
            y_score = probs.cpu().numpy()
            print(len(y_class))
            print(y_true.shape)
            print(y_score.shape)            
            breakpoint()

            logging_output["_y_true"] = y_true
            logging_output["_y_score"] = y_score
            logging_output["_y_class"] = y_class

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        batch_size = utils.item(
            sum(log.get("batch_size", 0) for log in logging_outputs)
        )
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )

        metrics.log_scalar(
            "loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3
        )

        metrics.log_scalar("batch_size", batch_size)

        if "_y_true" in logging_outputs[0] and "_y_score" in logging_outputs[0]:
            y_true = np.concatenate([log["_y_true"] for log in logging_outputs if "_y_true" in log])
            y_score = np.concatenate([log["_y_score"] for log in logging_outputs if "_y_score" in log])
            y_class = np.concatenate([log["_y_class"] for log in logging_outputs if "_y_class" in log])
            
            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, y_class)
