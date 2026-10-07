"""Train the authors' OCN on each TON-IoT held-out-attack partition.

Run from any directory:
    python baseline_wrappers/train_ocn.py

Inputs: no_<attack>/train_1_raw.csv and train_2_raw.csv, with a class column.
Outputs: results/ocn/ocn/ocn_no_<attack>.pt. Use --help for shorter trial runs.

This wrapper invokes train_pre and train_sharedcnn unchanged. It supplies the
missing pretraining stage and isolates their centroid file per experiment.
The final epoch is saved, without the authors' test-driven model selection.
Thresholds remain the original last-batch thresholds; no recalibration is done.
"""

import argparse
import math
from pathlib import Path
import tempfile
import time

import pandas as pd
import torch

if __package__:
    from . import utils_ocn as utils
else:
    import utils_ocn as utils


def train_partition(ocn, unknown_class, args):
    partition = args.data_dir / f"no_{unknown_class}"
    train_1 = utils.read_raw(partition / "train_1_raw_data.csv")
    train_2 = utils.read_raw(partition / "train_2_raw_data.csv")
    if set(train_1.columns) != set(train_2.columns):
        raise ValueError(f"Training feature schemas differ in {partition}.")
    frame = pd.concat([train_1, train_2], ignore_index=True)
    classes = sorted(frame["class"].unique().tolist())
    if unknown_class in classes:
        raise ValueError(f"Held-out attack {unknown_class!r} appears in the training data.")
    if len(classes) < 2:
        raise ValueError("The original OCN Fisher loss requires at least two known classes.")
    # The original loop processes len(loader)-1 batches, so it needs >=2 batches.
    if len(frame) <= args.batch_size:
        raise ValueError("Original OCN requires more training rows than --batch-size.")

    utils.seed_everything(args.seed)
    start = time.perf_counter()
    preprocessing = utils.fit_preprocessing(frame)
    features = utils.transform_features(frame, preprocessing)
    labels = frame["class"].map({name: i for i, name in enumerate(classes)}).to_numpy(dtype="int64")
    ocn.device = utils.resolve_device(args.device)
    ocn.BATCH_SIZE = args.batch_size
    model = ocn.models.SharedCNN(len(classes)).to(ocn.device)

    for epoch in range(1, args.pretrain_epochs + 1):
        ocn.train_pre(epoch, model, features, labels)
        utils.check_finite(model)

    # train_sharedcnn reads/writes this filename directly; keep it out of both
    # the authors' checkout and other experiments' artifacts.
    with tempfile.TemporaryDirectory(prefix="ocn-") as scratch:
        with utils.working_directory(scratch):
            torch.save(torch.zeros(len(classes), len(classes)), "CICIDS_centroids.pt")
            for epoch in range(1, args.epochs + 1):
                ocn.lamda = torch.tensor(0.05 * math.exp(-5 * epoch / args.epochs))
                ocn.alpha = torch.tensor(0.0001 * math.exp(-5 * epoch / args.epochs))
                ocn.beta = torch.tensor(0.01)
                thresholds = ocn.train_sharedcnn(
                    epoch, model, rank_rate=args.rank_rate,
                    max_threshold=torch.zeros(len(classes)),
                    data=features, label=labels, N_class=len(classes),
                ).detach().cpu()
                centroids = torch.load("CICIDS_centroids.pt", map_location="cpu", weights_only=True)
                utils.check_finite(model, centroids, thresholds)

    elapsed = round(time.perf_counter() - start, 2)
    checkpoint = {
        "format_version": 1,
        "unknown_class": unknown_class,
        "classes": classes,
        "preprocessing": preprocessing,
        "model_state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "centroids": centroids.detach().cpu(),
        "thresholds": thresholds,
        "training": {
            "epochs": args.epochs, "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size, "rank_rate": args.rank_rate,
            "seed": args.seed, "seconds": elapsed,
        },
    }
    output = args.results_dir / "ocn" / f"ocn_no_{unknown_class}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Saved {output}")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=utils.DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=utils.DEFAULT_RESULTS_DIR)
    parser.add_argument("--classes", nargs="+", choices=utils.CLASSES, default=list(utils.CLASSES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rank-rate", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a PyTorch device such as cuda:0")
    args = parser.parse_args()
    if args.epochs < 1 or args.pretrain_epochs < 1 or args.batch_size < 2:
        parser.error("epochs and pretrain-epochs must be >=1; batch-size must be >=2")
    if not 0 <= args.rank_rate < 1:
        parser.error("rank-rate must be in [0, 1)")
    args.data_dir = args.data_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    ocn = utils.load_ocn()
    print("Training OCNs...")
    times = {}
    for unknown_class in args.classes:
        print(f"\nHeld-out attack: {unknown_class}")
        times[unknown_class] = train_partition(ocn, unknown_class, args)
    print("Time taken for each class:")
    for name, elapsed in times.items():
        print(f"{name} {elapsed} s")


if __name__ == "__main__":
    main()
