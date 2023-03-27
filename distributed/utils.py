# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import logging
import random
import pickle
import socket
import struct
import warnings
from typing import Any, Optional, List
from dataclasses import dataclass

import torch
import torch.distributed as dist

import utils

logger = logging.getLogger(__name__)

def is_master(args):
    return args.distributed_rank == 0

def infer_init_method(args, force_distributed=False):
    if all(
        key in os.environ
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    ):
        _infer_torch_distributed_launch_init(args)
    elif args.distributed_world_size > 1 or force_distributed:
        _infer_single_node_init(args)

def _infer_torch_distributed_launch_init(args):
    args.distributed_init_method = "env://"
    args.distributed_world_size = int(os.environ["WORLD_SIZE"])
    args.distributed_rank = int(os.environ["RANK"])
    args.distributed_no_spawn = True

def _infer_single_node_init(args):
    assert (
        args.distributed_world_size <= torch.cuda.device_count()
    ), f"world size is {args.distributed_world_size} but have {torch.cuda.device_count()} available devices"
    port = random.randint(10000, 20000)
    args.distributed_init_method = "tcp://localhost:{port}".format(port=port)

def distributed_init(args):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        warnings.warn(
            "Distributed is already initialized, cannot initialize twice!"
        )
    else:
        logger.info(
            "disrtibuted init (rank {}): {}".format(
                args.distributed_rank,
                args.distributed_init_method
            )
        )
        dist.init_process_group(
            backend = "nccl",
            init_method = args.distributed_init_method,
            world_size = args.distributed_world_size,
            rank = args.distributed_rank
        )
        logger.info(
            "initialized host {} as rank {}".format(
                socket.gethostname(),
                args.distributed_rank
            )
        )

        #perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())
    
    args.distributed_rank = torch.distributed.get_rank()

    if is_master(args):
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    return args.distributed_rank

def distributed_main(i, main, args, kwargs):
    args.device_id = i
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
    if args.distributed_rank is None: # torch.multiprocessing.spawn
        args.distributed_rank = kwargs.pop("start_rank", 0) + i
    
    args.distributed_rank = distributed_init(args)

    after_distributed_init_fn = kwargs.pop("after_distributed_init_fn", None)
    if after_distributed_init_fn:
        args = after_distributed_init_fn(args)
    
    main(args, **kwargs)

    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())

def call_main(args, main, **kwargs):
    if args.distributed_init_method is None:
        infer_init_method(args)
    
    if args.distributed_init_method is not None:
        #distributed training
        start_rank =args.distributed_rank
        args.distributed_rank = None # assign automatically
        kwargs["start_rank"] = start_rank
        torch.multiprocessing.spawn(
            fn = distributed_main,
            args = (main, args, kwargs),
            nprocs = min(
                torch.cuda.device_count(),
                args.distributed_world_size
            ),
            join = True
        )
    else:
        # single GPU  main
        main(args, **kwargs)

def get_world_size(group):
    if torch.distributed.is_initialized():
        return dist.get_world_size(group = group)
    else:
        return 1

def get_rank(group):
    return dist.get_rank(group = group)

def get_global_group():
    if torch.distributed.is_initialized():
        if not hasattr(get_global_group, "_global_group"):
            # ideally we could use torch.distributed.group.WORLD, but it seems
            # to cause random NCCL hangs in some cases
            get_global_group._global_group = dist.new_group()
        return get_global_group._global_group
    else:
        return None

def get_global_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_global_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1

def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    return get_global_group()

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return get_rank(get_data_parallel_group())

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return get_world_size(get_data_parallel_group())

def all_reduce(tensor, group, op="sum"):
    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        raise NotImplementedError
    dist.all_reduce(tensor, op = op, group = group)
    return tensor

def all_gather_list(data, group = None, max_size = 16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable and any CUDA tensors will be moved
    to CPU and returned on CPU as well.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group: group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """

    if group is None:
        group = get_global_group()
    rank = get_rank(group = group)
    world_size = get_world_size(group = group)

    buffer_size = max_size * world_size
    if (
        not hasattr(all_gather_list, "_buffer")
        or all_gather_list._buffer.numel() < buffer_size
    ):
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    data = utils.move_to_cpu(data)
    enc = pickle.dumps(data)
    enc_size = len(enc)
    header_size = 4 # size of header that contains the length of the encoded data
    size = header_size + enc_size
    if size > max_size:
        raise ValueError(
            "encoded data size ({}) exceeds max_size ({})".format(size, max_size)
        )
    
    header = struct.pack(">I", enc_size)
    cpu_buffer[:size] = torch.ByteTensor(list(header + enc))
    start = rank * max_size
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group = group)

    buffer = buffer.cpu()
    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            (enc_size,) = struct.unpack(">I", bytes(out_buffer[:header_size].tolist()))
            if enc_size > 0:
                result.append(
                    pickle.loads(
                        bytes(out_buffer[header_size : header_size + enc_size].tolist())
                    )
                )
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data. "
            # "Try rerunning with --ddp-backend=legacy_ddp and see if that helps."
        )

def broadcast(tensor, src, group):
    dist.broadcast(tensor, src = src, group = group)

def broadcast_tensors(
    tensors: Optional[List[torch.Tensor]],
    src_rank: int,
    group: object,
    dist_device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    """
    Broadcast a list of tensors without other (non-src) ranks needing to know
    the dtypes/shapes of the tensors.
    """
    if dist_device is None:
        dist_device = torch.device("cuda")
    
    #share metadata first to simplify transfer
    is_src_rank = (get_rank(group) == src_rank)
    if is_src_rank:
        metadata = [
            {"size": t.size(), "dtype": t.dtype, "device": t.device} for t in tensors
        ]
        metadata = _broadcast_object_slow(metadata, src_rank, group, dist_device)
    else:
        metadata = _broadcast_object_slow(None, src_rank, group, dist_device)
    
    out_tensors = []
    for i, meta in enumerate(metadata):
        if is_src_rank:
            tensor = tensors[i]
            broadcast(tensors[i].to(dist_device), src = src_rank, group = group)
        else:
            tensor = torch.zeros(
                [meta["size"].numel()], dtype = meta["dtype"], device = dist.device
            )
            broadcast(tensor, src = src_rank, group = group)
        tensor = tensor.view(meta["size"]).to(meta["device"])
        out_tensors.append(tensor)
    
    return out_tensors

def broadcast_object(
    obj : Any,
    src_rank : int,
    group : object,
    dist_device: Optional[torch.device] = None
) -> Any:
    """Broadcast an arbitrary Python object to other workers."""
    if dist_device is None:
        dist_device = torch.device("cuda")
    
    if get_rank(group) == src_rank:
        # split thje tensors from the non-tensors so we can broadcast them
        # directly, avoiding unnecessary serialization/deserialization
        tensors = []
        obj = _split_tensors_from_obj(obj, tensors)
        obj = _broadcast_object_slow(obj, src_rank, group, dist_device)
    else:
        obj = _broadcast_object_slow(None, src_rank, group, dist_device)
        tensors = broadcast_tensors(None, src_rank, group, dist_device)
    return _put_tensors_in_obj(obj, tensors)

def _broadcast_object_slow(
    obj: Any, src_rank: int, group: object, dist_device: torch.device,
) -> Any:
    if get_rank(group) == src_rank:
        # Emit data
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        buffer = torch.ByteTensor(buffer.getbuffer()).to(dist_device)
        length = torch.LongTensor([len(buffer)]).to(dist_device)
        broadcast(length, src = src_rank, group = group)
        broadcast(buffer, src = src_rank, group = group)
    else:
        # Fetch from the source
        length = torch.LongTensor[0].to(dist_device())
        broadcast(length, src = src_rank, group = group)
        buffer = torch.ByteTensor(int(length.item())).to(dist_device)
        broadcast(buffer, src = src_rank, group = group)
        buffer = io.BytesIO(buffer.cpu().numpy())
        obj = torch.load(buffer, map_location = "cpu")
    return obj

@dataclass(frozen = True)
class _TensorPlaceholder:
    index: int

def _split_tensors_from_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if torch.is_tensor(obj):
        placeholder = _TensorPlaceholder(index = len(tensors))
        tensors.append(obj)
        return placeholder
    elif isinstance(obj, dict):
        return {k: _split_tensors_from_obj(v, tensors) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [_split_tensors_from_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_split_tensors_from_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_split_tensors_from_obj(v, tensors) for v in obj}
    else:
        return obj

def _put_tensors_in_obj(obj: Any, tensors: List[torch.Tensor]) -> Any:
    if isinstance(obj, _TensorPlaceholder):
        return tensors[obj.index]
    elif isinstance(obj, dict):
        return {k: _put_tensors_in_obj(v, tensors) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_put_tensors_in_obj(v, tensors) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_put_tensors_in_obj(v, tensors) for v in obj)
    elif isinstance(obj, set):
        return {_put_tensors_in_obj(v, tensors) for v in obj}
    else:
        return obj