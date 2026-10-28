"""Script to train CLOSR model"""
import random
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import numpy as np
import torch as T
import argparse
from baseline_wrappers.closr.model.model import CLOSRMLP
from baseline_wrappers.closr.losses.closr_loss import CLOSRLoss
import time
from baseline_wrappers.closr.utils_closr import load_partition
from baseline_wrappers.closr.data.loaders import tabular_dl
from baseline_wrappers.closr.util.schedules import WarmupCosineSchedule, LRSchedule
from baseline_wrappers.closr.util.meter import AverageMeter
from baseline_wrappers.closr.util.checkpoint import make_checkpoint

from baseline_wrappers.utils_baselines import DEFAULT_DATA_DIR, discover_classes, record_stage_time

def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    # data config
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="path to dataset"
    )
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parents[2] / "results/closr")
    parser.add_argument("--classes", nargs="+", help="Optional subset of discovered classes; default: all")

    # model config
    parser.add_argument(
        "--d_out", type=int, default=64, help="model output dimensionality"
    )
    parser.add_argument(
        "--neurons",
        type=str,
        default="1024,1024,1024",
        help="neurons in each mlp block",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument(
        "--residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use residual connections in mlp",
    )

    # loss config
    parser.add_argument(
        "--margin", type=float, default=1.0, help="loss function margin value"
    )
    parser.add_argument(
        "--squared",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to square distance metric in loss calculation",
    )

    # opt config
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0403709, help="weight decay"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--print_freq", type=int, default=100, help="how many batches to print after"
    )

    opt = parser.parse_args()
    if opt.epochs < 1 or opt.batch_size < 2 or opt.print_freq < 1:
        parser.error("epochs and print_freq must be positive; batch_size must be >=2")
    if not 0 <= opt.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if opt.device == "auto":
        opt.device = "cuda" if T.cuda.is_available() else "cpu"

    try:
        opt.classes = discover_classes(opt.data_dir, opt.classes)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    # parse neurons into list
    try:
        opt.neurons = [int(n) for n in opt.neurons.split(",")]
        if min(opt.neurons) < 1 or opt.d_out < 1:
            raise ValueError
    except ValueError:
        parser.error("neurons and d_out must be positive integers")

    return opt


def set_loader(opt, partition_path):
    x_train, y_train, _, _, _, _, opt.preprocessing = load_partition(
        data_path=partition_path, include_test=False,
    )
    opt.d_in = x_train.shape[1]
    opt.n_classes = len(opt.preprocessing["classes"])

    train_dl = tabular_dl(
        x=x_train,
        y=y_train,
        batch_size=min(opt.batch_size, len(y_train)),
        balanced=True,
        collate_fn=None,
        drop_last=True,
        num_workers=0,
    )

    return train_dl


def set_model(opt):
    # get model
    model = CLOSRMLP(
        d_in=opt.d_in,
        n_classes=opt.n_classes,
        d_out=opt.d_out,
        neurons=opt.neurons,
        dropout=opt.dropout,
        residual=opt.residual,
    )
    model = model.to(opt.device)

    # get loss
    criterion = CLOSRLoss(
        m=opt.margin,
        squared=opt.squared,
        n_classes=opt.n_classes,
    )
    return model, criterion


def set_optimiser(opt, model, train_dl):
    optimiser = T.optim.AdamW(
        model.parameters(),
        lr=1e-6,  # initial learning rate
        betas=(0.9, 0.999),
        weight_decay=opt.weight_decay,
    )

    base_schedule = WarmupCosineSchedule(
        start_val=1e-6,
        end_val=1e-6,
        ref_val=opt.lr,
        T_max=(opt.epochs * len(train_dl)),
        warmup_steps=int((opt.epochs // 10) * len(train_dl)),
    )

    lr_schedule = LRSchedule(
        optimiser,
        schedule=base_schedule,
    )

    return optimiser, lr_schedule


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    schedule,
    epoch,
    opt,
):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    frac_pos_vals = AverageMeter()

    end = time.time()
    for idx, (x, y) in enumerate(train_loader):
        data_time.update(time.time() - end)

        x = x.to(opt.device)
        y = y.to(opt.device)
        bsz = y.size(0)

        z = model(x)
        loss = criterion(
            x=z,
            y=y,
        )
        if not T.isfinite(loss):
            raise RuntimeError("CLOSR produced a nonfinite training loss")

        # update metric
        losses.update(loss.item(), bsz)
        frac_pos_vals.update(criterion.get_fraction_pos().item())

        # optimser
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "frac_pos {frac_pos_vals.val:.3f} ({frac_pos_vals.avg:.3f})".format(
                    epoch,
                    idx + 1,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    frac_pos_vals=frac_pos_vals,
                )
            )
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    with record_stage_time(opt.results_dir, "train"):
        print("Training CLOSR...")

        # Create "results" directory if it doesn't exist
        opt.results_dir.mkdir(parents=True, exist_ok=True)

        for cls in opt.classes:
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            T.manual_seed(opt.seed)

            partition_path = opt.data_dir / f"no_{cls}"
            # build data loader
            train_loader = set_loader(opt, partition_path)

            # build model and criterion
            model, criterion = set_model(opt)

            # build optimizer
            optimiser, schedule = set_optimiser(opt, model, train_loader)

            # training routine
            for epoch in range(1, opt.epochs + 1):
                # train for one epoch
                time1 = time.time()
                train(train_loader, model, criterion, optimiser, schedule, epoch, opt)
                time2 = time.time()
                print("epoch {}, total time {:.2f}".format(epoch, time2 - time1))

            os.makedirs(opt.results_dir / "closr", exist_ok=True)
            # save the trained model
            make_checkpoint(
                model=model,
                optimiser=optimiser,
                # Store scheduler state without pickling the scheduler object.
                schedular_state_dict=schedule.state_dict(),
                path=str(opt.results_dir / "closr" / f"closr_no_{cls}.pt.tar"),
                format_version=1,
                preprocessing=opt.preprocessing,
                model_config={key: getattr(opt, key) for key in
                              ("d_in", "n_classes", "d_out", "neurons", "dropout", "residual")},
                seed=opt.seed,
            )

if __name__ == "__main__":
    main()
