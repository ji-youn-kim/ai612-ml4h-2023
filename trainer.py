# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import math
import warnings
import logging
from typing import List, Dict, Any, Union
from itertools import chain
from argparse import Namespace

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import distributed.utils as distributed_utils
import utils
import checkpoint_utils
import models
import optim
from loggings import meters, metrics
from criterion import MultiTaskCriterion, SimCLRCriterion
from optim import lr_scheduler
from file_io import PathManager

logger = logging.getLogger(__name__)

class Trainer(object):
    """Main class for training."""
    
    def __init__(
        self,
        args: Namespace,
        model: models.BaseModel,
        criterion: Union[MultiTaskCriterion, SimCLRCriterion],
        train: torch.utils.data.Dataset = None,
        test: torch.utils.data.Dataset = None,
    ):
        self.args = args

        assert (
            (train is not None or test is not None)
            and (train is None or test is None)
        ), "train and test should not be activated together"

        valid_dataset = None
        if train is not None:
            if args.valid_percent > 0:
                train_percent = 1 - args.valid_percent

                subset_lengths: List[int] = []
                for i, frac in enumerate([train_percent, args.valid_percent]):
                    if frac < 0 or frac > 1:
                        raise ValueError(f"Fraction at index {i} is not between 0 and 1")
                    n_items_in_split = int(
                        math.floor(len(train) * frac)  # type: ignore[arg-type]
                    )
                    subset_lengths.append(n_items_in_split)
                remainder = len(train) - sum(subset_lengths)  # type: ignore[arg-type]
                # add 1 to all the lengths in round-robin fashion until the remainder is 0
                for i in range(remainder):
                    idx_to_add_at = i % len(subset_lengths)
                    subset_lengths[idx_to_add_at] += 1
                lengths = subset_lengths
                for i, length in enumerate(lengths):
                    if length == 0:
                        warnings.warn(f"Length of split at index {i} is 0. "
                                    f"This might result in an empty dataset.")

                dataset, valid_dataset = random_split(train, lengths)
            else:
                dataset = train
        elif test is not None:
            dataset = test

        collator = dataset.collator if valid_dataset is None else dataset.dataset.collator

        dummy_to_invoke_collator = [{"dummy": "dummy"}]
        try:
            collator(dummy_to_invoke_collator)
            use_collator = True
        except NotImplementedError:
            use_collator = False
        except Exception:
            use_collator = True

        self.iterator = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False if distributed_utils.get_data_parallel_world_size() > 1 else True,
            sampler=(
                DistributedSampler(dataset)
            ) if distributed_utils.get_data_parallel_world_size() > 1 else None,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collator if use_collator else None
        )
        self.valid_iterator = None
        if valid_dataset is not None:
            self.valid_iterator = DataLoader(
                dataset=valid_dataset,
                batch_size=args.batch_size,
                shuffle=False if distributed_utils.get_data_parallel_world_size() > 1 else True,
                sampler=(
                    DistributedSampler(valid_dataset)
                ) if distributed_utils.get_data_parallel_world_size() > 1 else None,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                collate_fn=collator if use_collator else None
            )

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._criterion = criterion
        self._model = model
        
        self._model = self._model.to(device=self.device)
        
        self.last_device = None
        
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        
        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None
        
        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group = distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)
        
        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change. """
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None
    
    @property
    def data_parallel_world_size(self):
        if self.args.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.args.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return self.data_parallel_world_size > 1

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        return self.is_data_parallel_master
    
    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if utils.has_parameters(self._criterion) and self.use_distributed_wrapper:
                self._wrapped_criterion = models.DistributedModel(
                    self.args,
                    self._criterion,
                    process_group=self.data_parallel_process_group,
                    device=self.device                    
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = models.DistributedModel(
                    self.args,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer
    
    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer() # this will initialize self._lr_scheduler
        return self._lr_scheduler
    
    def _build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                chain(self.model.parameters(), self.criterion.parameters())
            )
        )
        self._optimizer = optim.build_optimizer(self.args, params)
        
        # we should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.args,
            self.optimizer
        )
        self._lr_scheduler.step_update(0)
    
    def state_dict(self):
        state_dict = {
            "args": self.args,
            "model": self.model.state_dict(),
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion)
                else None
            ),
            "optimizer_history": (self._optim_history or [])
            + [
                {
                    "criterion_name": self.get_criterion().__class__.__name__,
                    "optimizer_name": self.optimizer.__class__.__name__,
                    "lr_scheduler_state": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else {},
                    "num_updates": self.get_num_updates()
                }
            ],
            "extra_state": {
                "previous_training_time": self.cumulative_training_time()
            }
        }
        
        if not self.args.no_save_optimizer_state:
            state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        return state_dict
    
    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=False
            )
        logger.info(f"Finished saving checkpoint to {filename}")
    
    # def load_checkpoint(
    #     self,
    #     filename,
    #     reset_optimizer=False,
    #     reset_lr_scheduler=False,
    #     optimizer_overrides=None,
    #     reset_meters=False
    # ):
    #     """
    #     Load all training state from a checkpoint file.
    #     rank = 0 will load the checkpoint, and then broadcast it to all
    #     other ranks.
    #     """
    #     extra_state, self._optim_history, last_optim_state = None, [], None

    #     logger.info(f"Preparing to load checkpoint {filename}")
    #     is_distributed = self.data_parallel_world_size > 1
    #     bexists = PathManager.isfile(filename)
    #     if bexists:
    #         load_on_all_ranks = self.args.load_checkpoint_on_all_dp_ranks
            
    #         if load_on_all_ranks or self.data_parallel_rank == 0:
    #             state = checkpoint_utils.load_checkpoint_to_cpu(
    #                 filename, load_on_all_ranks=load_on_all_ranks
    #             )
    #             last_optim_state = state.get("last_optimizer_state", None)
                
    #             if (
    #                 not load_on_all_ranks
    #                 and "last_optimizer_state" in state
    #                 and is_distributed
    #             ):
    #                 state["last_optimizer_state"] = "SHARDED"
    #         else:
    #             last_optim_state = None
    #             state = None
            
    #         if is_distributed and not load_on_all_ranks:
    #             state = distributed_utils.broadcast_object(
    #                 state,
    #                 src_rank=0,
    #                 group=self.data_parallel_process_group,
    #                 dist_device=self.device
    #             )
    #             if self.data_parallel_rank > 0:
    #                 last_optim_state = state.get("last_optimizer_state", None)
            
    #         # load model parameters
    #         try:
    #             self.model.load_state_dict(state["model"], strict=True)
    #             # save memory for later steps
    #             del state["model"]
    #             if utils.has_parameters(self.get_criterion()):
    #                 self.get_criterion().load_state_dict(
    #                     state["criterion"], strict=True
    #                 )
    #                 del state["criterion"]
    #         except Exception:
    #             raise Exception(
    #                 f"Cannot load model parameters from checkpoint {filename};"
    #                 "please ensure that the architectures matches."
    #             )
    #         extra_state = state["extra_state"]
    #         self._optim_history = state["optimizer_history"]
        
    #     if last_optim_state is not None and not reset_optimizer:
    #         # rebuild optimizer after loading model, since params may have changed
    #         self._build_optimizer()

    #         # only reload optimizer and lr_scheduler if they match
    #         last_optim = self._optim_history[-1]
    #         assert (
    #             last_optim["criterion_name"] == self.get_criterion().__class__.__name__
    #         ), f"Criterion does not match; please reset the optimizer (--reset-optimizer). {last_optim['criterion_name']} vs {self.get_criterion().__class__.__name__}"
    #         assert (
    #             last_optim["optimzier_name"] == self.optimizer.__class__.__name__
    #         ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__}"

    #         if not reset_lr_scheduler:
    #             self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

    #         if not load_on_all_ranks and is_distributed:
    #             last_optim_state = self.optimizer.broadcast_global_state_dict(
    #                 last_optim_state
    #             )

    #         self.optimzier.load_state_dict(last_optim_state, optimizer_overrides)
    #         self.set_num_updates(last_optim["num_updates"])



    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))
        
        self.lr_step_begin_epoch(epoch)

    def begin_valid_epoch(self, epoch):
        """Called at the beginning of each validation epoch."""
        pass

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight = 0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight = 0, priority = 300)
        return new_lr

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model
    
    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    @metrics.aggregate("train")
    def train_step(self, sample):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        self.state_dict()
        
        metrics.log_start_time("train_wall", priority=800, round=0)

        # forward and backward pass
        sample = self._prepare_sample(sample)

        self.model.set_num_updates(self.get_num_updates())
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = self.criterion(self.model, sample)
        with torch.autograd.profiler.record_function("backward"):
            self.optimizer.backward(loss)
        del loss
        
        # emptying the CUDA cache after the first step can
        # reduce the chance of OOM
        if self.cuda and self.get_num_updates() == 0:
            torch.cuda.empty_cache()

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        logging_outputs = [logging_output]
        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            logging_outputs, (
                sample_size, total_train_time
            ) = self._aggregate_logging_outputs(logging_outputs, sample_size, train_time)
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        with torch.autograd.profiler.record_function("reduce-grads"):
            # reduce gradients across workers
            self.optimizer.all_reduce_grads(self.model)
            if utils.has_parameters(self.criterion):
                self.optimizer.all_reduce_grads(self.criterion)
        
        with torch.autograd.profiler.record_function("multiply-grads"):
            # multiply gradients by (data_parallel_size / sample_size) since
            # DDP normalizes by the number of data parallel workers for
            # improved fp16 precision.
            # Thus we get (sum_of_gradients / sample_size) at the end.
            # In case of fp16, this step also undoes loss scaling.
            # (Debugging note: Some optimizers perform this scaling on the
            # fly, so inspecting model.parameters() or optimizer.params may
            # still show the original, unscaled gradients.)
            numer = (
                self.data_parallel_world_size
                if self._sync_stats()
                else 1
            )
            self.optimizer.multiply_grads(numer / (sample_size or 1.0))
            # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
            # way that avoids CPU/device transfers in case sample_size is a GPU or
            # TPU object. The assumption is that the gradient itself is also 0.

        with torch.autograd.profiler.record_function("clip-grads"):
            # clip grads
            grad_norm = self.clip_grad_norm(self.args.clip_norm)

        if not torch.isfinite(grad_norm).all():
            # check local gradnorm single GPU case
            raise FloatingPointError("gradients are Nan/Inf")

        with torch.autograd.profiler.record_function("optimizer"):
            # take an optimization step
            self.optimizer.step()

        logging_output = None
        self.set_num_updates(self.get_num_updates() + 1)
        
        if self.cuda and self.cuda_env is not None:
            # log minimum free memory over the iteration
            gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            gb_free = self.cuda_env.total_memory_in_GB - gb_used
            metrics.log_scalar(
                "gb_free", gb_free, priority=1500, round=1, weight=0
            )
        
        # extract private logs (usually only for valid step) before logging
        logging_outputs = list(map(
            lambda x: {key: x[key] for key in x if not key.startswith("_")}, logging_outputs)
        )
        # log stats
        logging_output = self._reduce_and_log_stats(
            logging_outputs, sample_size, grad_norm
        )

        # clear CUDA cache to reduce memory fragmentation
        if (
            self.cuda
            and self.args.empty_cache_freq > 0
            and (
                (self.get_num_updates() + self.args.empty_cache_freq - 1)
                % self.args.empty_cache_freq
            )
            == 0
        ):
            torch.cuda.empty_cache()
        
        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            
            sample = self._prepare_sample(sample)
            loss, sample_size, logging_output = self.criterion(self.model, sample)

            logging_outputs = [logging_output]
        
        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size, ) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size
            )
        
        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
        
        return loss, sample_size, logging_output

    def post_validate(self, log_output, agg, num_updates, **kwargs):
        for key in agg.keys():
            if key.startswith("_") and key.endswith("auc"):
                log_output[key[1:-3] + "auroc"] = agg[key].auroc
                log_output[key[1:-3] + "auprc"] = agg[key].auprc

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):        
        return self.optimizer.clip_grad_norm(
            clip_norm, aggregate_norm_fn=None
        )

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
    ):
        return self._all_gather_list_sync(logging_outputs, *extra_stats_to_sum)

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size = getattr(self.args, "all_gather_list_size", 1048576),
                    group = self.data_parallel_process_group
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fp_convert_sample(self, sample):
        def apply_float(t):
            if t.dtype in [torch.float64, torch.float32, torch.int16]:
                return t.to(dtype = torch.float)
            return t

        sample = utils.apply_to_sample(apply_float, sample)
        
        return sample

    def _prepare_sample(self, sample):
        # Given that PCIe/NVLink bandwith is significantly smaller than DRAM bandwidth
        # it makes sense to do the format conversion on the CPU and then transfer
        # a smaller buffer to the device. This also saves GPU memory capacity.
        
        if self.cuda:
            sample = utils.move_to_cuda(sample)

        sample = self._fp_convert_sample(sample)
        
        return sample

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm = None):
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.args.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.args.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.get_criterion().__class__.reduce_metrics(logging_outputs)
                del logging_outputs
            
            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]
            return logging_output