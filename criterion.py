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
from loggings.meters import safe_round

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
        net_output = model(**sample)
        
        logits = model.get_logits(net_output).float() # (batch, 52)
        target = model.get_targets(sample) # (batch, 28)

        assert logits.dim() == 2 and target.dim() == 2

        reduction = "none" if not reduce else "mean"
        sample_size = 1

        idx = 22
        binary_target = target[:, :idx]
        binary_logits = logits[:, :idx]

        loss = F.binary_cross_entropy_with_logits(
            binary_logits,
            binary_target.float(),
            reduction=reduction
        )
        ##### CODE ADDED #####
        # if loss.isnan().any():
        #     breakpoint()
        ######################

        num_classes = [6, 6, 5, 5, 5, 3]
        for i, num_class in enumerate(num_classes, start=idx):
            multi_class_target = target[:, i]
            multi_class_logits = logits[:, idx: idx + num_class]
            multi_class_loss = F.cross_entropy(
                input=multi_class_logits,
                target=multi_class_target,
                reduction=reduction,
                ignore_index=-1
            )
            ##### CODE ADDED #####
            if multi_class_loss.isnan().any():
                loss += 0
            else:
                loss += multi_class_loss
            # if loss.isnan().any():
            #     breakpoint()
            ######################
            idx += num_class

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "batch_size": len(target),
            "sample_size": sample_size
        }

        with torch.no_grad():
            idx = 22

            y_class = np.array(sum([[i for _ in range(len(target))] for i in range(idx)], []))
            y_true = target.T[:idx].flatten().cpu().numpy()
            y_score = torch.sigmoid(logits).T[:idx].flatten().cpu().numpy()

            logging_output["binary_y_true"] = y_true
            logging_output["binary_y_score"] = y_score
            logging_output["binary_y_class"] = y_class

            for i, num_class in enumerate(num_classes, start=idx):
                y_true = target[:, i].cpu().numpy()
                multiclass_logits = logits[:, idx: idx + num_class]
                multiclass_y_score = torch.softmax(multiclass_logits, dim=-1).cpu().numpy()

                activated_idx = np.where(y_true != -1)
                y_true = y_true[activated_idx]
                multiclass_y_score = multiclass_y_score[activated_idx]
                logging_output[f"multiclass_y_true_{i}"] = y_true
                logging_output[f"multiclass_y_score_{i}"] = multiclass_y_score

                idx += num_class

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

        if "binary_y_true" in logging_outputs[0] and "binary_y_score" in logging_outputs[0]:
            y_true = np.concatenate(
                [log["binary_y_true"] for log in logging_outputs if "binary_y_true" in log]
            )
            y_score = np.concatenate(
                [log["binary_y_score"] for log in logging_outputs if "binary_y_score" in log]
            )
            y_class = np.concatenate(
                [log["binary_y_class"] for log in logging_outputs if "binary_y_class" in log]
            )
            
            metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, y_class)

        builtin_keys = {
            "loss",
            "batch_size",
            "sample_size",
            "binary_y_true",
            "binary_y_score"
            "binary_y_class"
        }

        for k in logging_outputs[0]:
            if k not in builtin_keys:
                if k.startswith("multiclass"):
                    y_true = np.concatenate(
                        [
                            log["multiclass_y_true_" + k[-2:]] for log in logging_outputs
                            if "multiclass_y_true_" + k[-2:] in log
                        ]
                    )
                    y_score = np.concatenate(
                        [
                            log["multiclass_y_score_" + k[-2:]] for log in logging_outputs
                            if "multiclass_y_score_" + k[-2:] in log
                        ]
                    )
                    builtin_keys.add("multiclass_y_true_" + k[-2:])
                    builtin_keys.add("multiclass_y_score_" + k[-2:])
                    
                    metrics.log_custom(meters.AUCMeter, "_auc", y_score, y_true, cls=int(k[-2:]), multiclass=True)


class SimCLRCriterion(_Loss):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.temp = 0.1
    
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
        net_output = model(**sample)

        logits = model.get_logits(net_output).float()

        logits = F.normalize(logits, dim=1)
        bsz = int(logits.shape[0] / 2)

        metrics.log_scalar("batch_size", bsz)
        metrics.log_scalar("emb_size", logits.shape[1])

        mask = 1 - torch.eye(bsz * 2, dtype=torch.uint8).to(logits.device)
        pos_ind = (
            torch.arange(bsz * 2).to(logits.device),
            2 * torch.arange(bsz, dtype=torch.long).unsqueeze(1).repeat(
                1, 2).view(-1, 1).squeeze().to(logits.device)
        )
        neg_mask = torch.ones((bsz * 2, bsz * 2 - 1), dtype=torch.uint8).to(logits.device)
        neg_mask[pos_ind] = 0
        
        # Cosine similarity computation
        sim_matrix = torch.matmul(logits, logits.T) # cosine similarity computation
        # Eliminate similarity between same view
        sim_matrix = torch.masked_select(sim_matrix, mask.bool()).view(sim_matrix.size(0), -1)

        positives = sim_matrix[pos_ind].unsqueeze(1)
        negatives = torch.masked_select(sim_matrix, neg_mask.bool()).view(sim_matrix.size(0), -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temp # divide by softmax temperature

        target = torch.zeros((logits.size(0), ), dtype=torch.long).to(logits.device)

        reduction = "none" if not reduce else "sum"

        loss = F.cross_entropy(logits, target, reduction=reduction)

        sample_size = logits.shape[0]

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "sample_size": sample_size
        }

        with torch.no_grad():
            if logits.numel() == 0: # in the case of no logits returned
                corr = 0
                count = 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0

                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = float(max.numel())
            
            logging_output["correct"] = corr
            logging_output["count"] = count
        
        return loss, sample_size, logging_output
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None: 
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            'loss', loss_sum / (sample_size or 1) / math.log(2), sample_size, round = 3
        )

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        metrics.log_scalar('correct', correct)

        total = sum(log.get("count", 0) for log in logging_outputs)
        metrics.log_scalar('total', total)

        if total > 0:
            metrics.log_derived( 
                'acc',
                lambda meters: safe_round(
                    meters['correct'].sum / meters['total'].sum, 5
                )
                if meters['total'].sum > 0
                else float("nan")
            )