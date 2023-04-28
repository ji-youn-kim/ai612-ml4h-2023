# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import collections
import logging
import traceback

import torch

from file_io import PathManager

logger = logging.getLogger(__name__)

def save_checkpoint(args, trainer, epoch, val_loss):
    from loggings import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(args.save_dir, exist_ok = True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        if args.pretrain:
            best_function = min
        else:
            best_function = max 
        save_checkpoint.best = best_function(val_loss, prev_best)

    if not trainer.should_save_checkpoint_on_current_rank:
        return
    
    write_timer = meters.StopwatchMeter()
    write_timer.start()

    updates = trainer.get_num_updates()

    logger.info(f"preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(args, a, b):
        if args.pretrain:
            return a <= b
        else:
            return a >= b
    
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}.pt".format(epoch)] = (epoch % args.save_interval == 0)
    checkpoint_conds["checkpoint_best.pt"] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(args, val_loss, save_checkpoint.best)
    )
    checkpoint_conds["checkpoint_last.pt"] = True

    extra_state = {"val_loss": val_loss}
    if hasattr(save_checkpoint, "best"):
        extra_state.update({"best": save_checkpoint.best})
    
    checkpoints = [
        os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond
    ]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            assert PathManager.copy(
                checkpoints[0], cp, overwrite=True
            ), f"Failed to copy {checkpoints[0]} to {cp}"
    
        write_timer.stop()
        logger.info(
            "Save checkpoint {} (epoch {} @ {} updates, score {}) (writing took {} seconds)".format(
                checkpoints[0], epoch, updates, val_loss, write_timer.sum
            )
        )
    
    # remove old epoch checkpoints; checkpoints are sorted in descending order
    checkpoints = checkpoint_paths(args.save_dir, pattern=r"checkpoint(\d+)\.pt")
    for old_chk in checkpoints[1:]:
        if os.path.lexists(old_chk):
            os.remove(old_chk)

def load_checkpoint_to_cpu(path, arg_overrides = None, load_on_all_ranks = False):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility).

    If doing single-GPU training or if the checkpoint is only being loaded by at
    most one process on each node (current default behavior is for only rank 0
    to read the checkpoint from disk), load_on_all_ranks should be False to
    avoid erros from torch.distributed not having been initialized or
    torch.distributed.barrier() hanging.

    If all processes on each node may be loading the checkpoint
    simultaneously, load_no_all_ranks should be set to True to avoid I/O
    conflicts.

    There's currently no support for > 1 but < all process loading the
    checkpoint on each node.
    """
    local_path = PathManager.get_local_path(path)
    # The locally cached file returned by get_local_path() may be stable for
    # remote files that are periodically updated/overwritten (ex:
    # checkpoint_last.pt) - so we remove the local copy, sync across processes
    # (if needed), and then download a fresh copy.
    if local_path != path and PathManager.path_requires_pathmanager(path):
        try:
            os.remove(local_path)
        except FileNotFoundError:
            # With potentially multiple processes removing the same file, the
            # file being missing is benign (missing_ok isn't available until
            # Python 3.8).
            pass
        if load_on_all_ranks:
            torch.distributed.barrier()
        local_path = PathManager.get_local_path(path)
    
    with open(local_path, "rb") as f:
        state = torch.load(f, map_location = torch.device("cpu"))

    if "args" in state and state["args"] is not None and arg_overrides is not None:
        args = state["args"]
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    
    return state

def checkpoint_paths(path, pattern = r"checkpoint(\d+)\.pt", keep_match = False):
    """Retrives all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order
    """
    pt_regexp = re.compile(pattern)
    files = PathManager.ls(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = float(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    if keep_match:
        return [(os.path.join(path, x[1]), x[0]) for x in sorted(entries, reverse = True)]
    else:
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse = True)]

def torch_persistent_save(obj, filename, async_write: bool = False):
    if async_write:
        with PathManager.opena(filename, "wb") as f:
            _torch_persistent_save(obj, f)
    else:
        if PathManager.supports_rename(filename):
            # do atomic save
            with PathManager.open(filename + ".tmp", "wb") as f:
                _torch_persistent_save(obj, f)
            PathManager.rename(filename + ".tmp", filename)

def _torch_persistent_save(obj, f):
    if isinstance(f, str):
        with PathManager.open(f, "wb") as h:
            torch_persistent_save(obj, h)
        return
    for i in range(3):
        try:
            return torch.save(obj, f)
        except Exception:
            if i == 2:
                logger.error(traceback.format_exc())

def verify_checkpoint_directory(save_dir: str) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.join(save_dir, "dummy")
    try:
        with open(temp_file_path, "w"):
            pass
    except OSError as e:
        logger.warning(
            "Unable to access checkpoint save directory: {}".format(save_dir)
        )
        raise e
    else:
        os.remove(temp_file_path)
