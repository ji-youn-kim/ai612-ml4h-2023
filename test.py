import os
import sys
import argparse
import logging
import logging.config
import time
from typing import Dict, List, Optional, Any

import torch

import models
import utils
import checkpoint_utils
import data
from loggings import metrics, meters, progress_bar
from criterion import MultiTaskCriterion
from distributed import utils as distributed_utils
from trainer import Trainer

# We need to setup root logger before importing any project libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger("test")

def get_parser():
    parser = argparse.ArgumentParser()
    
    # required arguments
    parser.add_argument(
        "--student_number",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        required=True
    )

    # test options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--pin_memory",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--all_gather_list_size",
        default=1048576,
        help="number of bytes reserved for gathering stats from workers"
    )

    # logging
    parser.add_argument(
        "--log_file",
        type=str,
        default=None
    )

    # wandb logging
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights and Biases entity(team) name to use for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights and Biases project name to use for logging"
    )
    
    return parser

def main(args: argparse.Namespace) -> None:
    overrides = vars(args)

    logger.info(f"Loading a state from {args.eval_path}")
    st = time.time()
    state = checkpoint_utils.load_checkpoint_to_cpu(args.eval_path, arg_overrides=overrides)
    
    saved_args = state["args"]

    set_struct(saved_args)

    model = models.build_model(saved_args.student_number + "_model", **vars(saved_args))
    model.load_state_dict(state["model"], strict=True)
    elapsed = time.time() - st
    logger.info(f"Loaded a checkpoint in {elapsed:.2f}s")

    criterion = MultiTaskCriterion.build_criterion(saved_args)

    logger.info(model)
    logger.info("model: {}".format(model.__class__.__name__))

    # load dataset
    if not hasattr(saved_args, "dataset"):
        saved_args.dataset = saved_args.student_number + "_dataset"
    dataset = data.build_dataset(saved_args)

    trainer = Trainer(saved_args, model, criterion, test=dataset)
    
    # Print args
    logger.info(saved_args)

    validate(saved_args, trainer, epoch=saved_args.max_epoch)

def validate(
    args: argparse.Namespace,
    trainer: Trainer,
    epoch: int,
) -> List[Optional[float]]:
    """Evaluate the model on the validation set and return the losses"""
    
    trainer.begin_valid_epoch(epoch)
    logger.info(f'begin validation on test subset')

    progress = progress_bar.progress_bar(
        trainer.iterator,
        log_format=None,
        log_file=args.log_file,
        log_interval=args.log_interval,
        epoch=epoch,
        tensorboard_logdir=None,
        default_log_format=("tqdm"),
        wandb_project=(
            args.wandb_project
            if distributed_utils.is_master(args)
            else None
        ),
        wandb_entity=(
            args.wandb_entity
            if distributed_utils.is_master(args)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(args.save_dir)
        ),
        azureml_logging=False
    )
    
    with metrics.aggregate(new_root=True) as agg:
        for i, sample in enumerate(progress):
            trainer.valid_step(sample)
    
    # log validation stats
    stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
    
    if hasattr(trainer, "post_validate"):
        trainer.post_validate(
            log_output=stats,
            agg=agg,
            num_updates=trainer.get_num_updates()
        )
    
    progress.print(stats, tag="test", step=trainer.get_num_updates())

    return stats["auroc"]

def get_valid_stats(
    args: argparse.Namespace,
    trainer: Trainer,
    stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()

    return stats

def set_struct(args):
    from datetime import datetime
    from pytz import timezone
    
    now = datetime.now()
    now = now.astimezone(timezone("Asia/Seoul"))
    
    root = os.path.abspath(
        os.path.dirname(__file__)
    )
        
    output_dir = os.path.join(
        root, "outputs", now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    os.chdir(output_dir)
    
    job_logging_cfg = {
        "version": 1,
        "formatters": {
            "simple": {
                "format": "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler", "formatter": "simple", "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.FileHandler", "formatter": "simple", "filename": "test.log"
            }
        },
        "root": {
            "level": "INFO", "handlers": ["console", "file"]
        },
        "disable_existing_loggers": False
    }
    logging.config.dictConfig(job_logging_cfg)
    
    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.mkdir(args.save_dir)
    
    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in vars(args).items():
            print("{}: {}".format(k, v), file=f)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)