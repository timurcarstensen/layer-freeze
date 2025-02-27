import argparse
import logging
import os
import time
from functools import partial
from pathlib import Path

import git
import neps
import torch
import torch.nn as nn
import torch.optim as optim
from neps.plot.tensorboard_eval import tblogger

from layer_freeze.utils import freeze_layers

from .utils import create_model, data_prep, full_fidelity_training, validate


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 10,
    # n_unfrozen_layers: int = 1,
) -> dict:
    # Training loop
    _start = time.time()
    forward_times = []
    backward_times = []
    model.train()
    step = 0
    for _ in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):  # noqa: B007
            step += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            forward_start = time.time()
            logits = model(data)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)

            loss = criterion(logits, target)

            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            # print statistics
            running_loss += loss.item()
        # print(f'[epoch={epoch},batch={batch_idx+1:<5d}]\tloss: {running_loss / (batch_idx+1):.6f}')
        training_loss_for_epoch = running_loss / (batch_idx + 1)
    _end = time.time()

    memory_used = torch.cuda.max_memory_allocated() / (1024 * 1024)

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    return {
        "loss": training_loss_for_epoch,
        "cost": _end - _start,
        "info_dict": {
            "train_loss": training_loss_for_epoch,
            "gpu_memory_used_mb": memory_used,
            "avg_forward_time_ms": avg_forward_time * 1000,
            "avg_backward_time_ms": avg_backward_time * 1000,
        },
    }


def training_pipeline(
    pipeline_directory: str,
    previous_pipeline_directory: str | None,
    epochs: int = 10,
    n_unfrozen_layers: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 0.008,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer_name: str = "adam",
    log_tensorboard: bool = True,
) -> dict:
    """Main training interface for HPO."""
    # init logger
    logger = tblogger()
    logger.initialize_writers()
    # reset memory stats
    torch.cuda.reset_peak_memory_stats()

    # Prepare data
    trainloader, validloader, _, num_classes = data_prep(batch_size=batch_size)

    # Define model with new parameters
    model = create_model(num_classes=num_classes)

    # freeze layers
    freeze_layers(model=model, n_unfrozen_layers=n_unfrozen_layers)
    logging.info(f"Model total num parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Model num parameters to train: "
        f"{sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))}"
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            momentum=beta1,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if torch.cuda.is_available():
        model = model.cuda()

    train_results = train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=trainloader,
        epochs=epochs,
    )
    val_err, val_cost = validate(model=model, val_loader=validloader)

    full_fidelity_results = full_fidelity_training(
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        optimizer=optimizer_name,
    )

    # Logging
    if log_tensorboard:
        tblogger.log(
            loss=val_err,
            current_epoch=epochs,
            write_summary_incumbent=True,
            writer_config_scalar=True,
            writer_config_hparam=True,
            extra_data={
                "train_loss": tblogger.scalar_logging(train_results["loss"]),
                "val_err": tblogger.scalar_logging(val_err),
                "n_trainable_params": tblogger.scalar_logging(
                    sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
                ),
                "n_total_params": tblogger.scalar_logging(
                    sum(p.numel() for p in model.parameters())
                ),
                "n_unfrozen_layers": tblogger.scalar_logging(n_unfrozen_layers),
                "perc_trainable_params": tblogger.scalar_logging(
                    (
                        sum(
                            p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())
                        )
                        / sum(p.numel() for p in model.parameters())
                    )
                    * 100
                ),
                "gpu_memory_used_mb": tblogger.scalar_logging(
                    train_results["info_dict"]["gpu_memory_used_mb"]
                ),
                "avg_forward_time_ms": tblogger.scalar_logging(
                    train_results["info_dict"]["avg_forward_time_ms"]
                ),
                "avg_backward_time_ms": tblogger.scalar_logging(
                    train_results["info_dict"]["avg_backward_time_ms"]
                ),
                "full_fidelity_val_acc": tblogger.scalar_logging(full_fidelity_results["val_acc"]),
                "full_fidelity_val_err": tblogger.scalar_logging(full_fidelity_results["val_err"]),
                "full_fidelity_cost": tblogger.scalar_logging(full_fidelity_results["cost"]),
            },
        )

    return {
        "loss": val_err,
        "cost": train_results["cost"],
        "info_dict": {
            "train_loss": train_results["loss"],
            "validation_time": val_cost,
            "current_epoch": epochs,
            "gpu_memory_used_mb": train_results["info_dict"]["gpu_memory_used_mb"],
            "pid": os.getpid(),
            "full_fidelity_results": full_fidelity_results,
            "n_trainable_params": sum(
                p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())
            ),
            "n_total_params": sum(p.numel() for p in model.parameters()),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--cpus_per_node", type=int, default=1, help="Number of cpus per node")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per node")
    parser.add_argument("--group_name", type=str, default="", help="Group name")
    parser.add_argument(
        "--n_unfrozen_layers", type=int, default=1, help="Number of layers to freeze"
    )
    args = parser.parse_args()

    # Count number of layers in Net by checking children
    model = create_model()
    num_layers = len(list(model.children()))
    if args.n_unfrozen_layers > num_layers:
        raise ValueError(
            f"n_unfrozen_layers ({args.n_unfrozen_layers}) must be <= {num_layers}, "
            f"the total number of layers in Net"
        )

    pipeline_space = {
        "learning_rate": neps.Float(1e-5, 1e-1, log=True, default=0.008),
        "beta1": neps.Float(0.9, 0.999, log=True, default=0.9),
        "beta2": neps.Float(0.9, 0.999, log=True, default=0.999),
        "weight_decay": neps.Float(1e-5, 0.1, log=True, default=0.01),
        "optimizer_name": neps.Categorical(["adam", "sgd"], default="adam"),
        "dropout_rate": neps.Float(0.0, 0.8, default=0.0),
    }

    root_directory = Path(git.Repo(".", search_parent_directories=True).working_tree_dir) / "output"
    if not root_directory.exists():
        try:
            root_directory.mkdir(parents=True)
        except FileExistsError:
            print("Directory already exists")

    algo = "random_search"
    output_tree = (
        f"{args.nodes}nodes_{args.cpus_per_node}cpus_"
        f"{args.gpus_per_node}gpus_{args.n_unfrozen_layers}unfrozen"
    )

    # TODO: set seed for reproducibility across torch and numpy
    neps.run(
        pipeline_space=pipeline_space,
        run_pipeline=partial(
            training_pipeline,
            log_tensorboard=True,
            n_unfrozen_layers=args.n_unfrozen_layers,
        ),
        searcher=algo,
        max_evaluations_total=250,
        root_directory=(f"{root_directory}/{args.group_name}/{algo}/{output_tree}/"),
        overwrite_working_directory=False,
    )
