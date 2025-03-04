import argparse
import logging
import time
from functools import partial
from pathlib import Path

import git
import neps
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.models import ResNet

from experiments.resnet.utils import create_model, data_prep_c100, validate
from layer_freeze.model_agnostic_freezing import FrozenModel


def training_pipeline(
    pipeline_directory: str,
    previous_pipeline_directory: str | None,
    epochs: int = 1,
    n_trainable_layers: int = 1,
    batch_size: int = 1024,
    learning_rate: float = 0.008,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer_name: str = "adam",
) -> dict:
    """Main training interface for HPO."""
    wandb.init(project="layer-freeze", group=f"{n_trainable_layers}_trainable", reinit=True)
    wandb.config.update(locals())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    trainloader, validloader, _, num_classes = data_prep_c100(
        batch_size=batch_size, dataloader_workers=8
    )

    # Define model with new parameters
    model = create_model(num_classes=num_classes)

    # freeze layers
    model = FrozenModel(
        n_trainable=n_trainable_layers, base_model=model, print_summary=False, unwrap=ResNet
    )

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    match optimizer_name.lower():
        case "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=weight_decay,
            )
        case "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=learning_rate,
                momentum=beta1,
                weight_decay=weight_decay,
            )
        case _:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    model = model.to(device)

    # Training loop
    _start = time.time()
    forward_times = []
    backward_times = []
    losses = []
    model.train()
    step = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, target) in enumerate(trainloader):  # noqa: B007
            step += 1
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = data.to(device)
            target = target.to(device)

            forward_start = time.time()
            outputs = model(data)
            forward_times.append(time.time() - forward_start)

            loss = criterion(outputs, target)
            wandb.log({"train_loss": loss.cpu().item()})
            losses.append(loss.cpu().item())
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_times.append(time.time() - backward_start)

            # print statistics
            running_loss += loss.item()
        wandb.log({"epoch": epoch})
        training_loss_for_epoch = running_loss / (i + 1)
        # TODO: log training curve
    _end = time.time()

    memory_used = torch.cuda.memory_allocated(device=device) / (1024**2)

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    # Validation loop
    val_err, val_time = validate(model, validloader, device)

    extra = {
        "validation_time": val_time,
        "current_epoch": epochs,
        "gpu_memory_used_mb": memory_used,
        "avg_forward_time_ms": avg_forward_time * 1000,
        "avg_backward_time_ms": avg_backward_time * 1000,
        "n_trainable_params": sum(
            p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())
        ),
        "n_total_params": sum(p.numel() for p in model.parameters()),
    }
    wandb.config.perc_trainable_params = (
        sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
        / sum(p.numel() for p in model.parameters())
    ) * 100
    wandb.log(extra)
    wandb.finish()

    return {
        "objective_to_minimize": val_err,
        "cost": _end - _start,
        "learning_curve": losses,
        "extra": extra,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes to use")
    parser.add_argument("--cpus_per_node", type=int, default=1, help="Number of cpus per node")
    parser.add_argument("--gpus_per_node", type=int, default=1, help="Number of gpus per node")
    parser.add_argument("--group_name", type=str, default="", help="Group name")
    parser.add_argument(
        "--n_trainable_layers", type=int, default=1, help="Number of layers to train"
    )
    args = parser.parse_args()

    pipeline_space = {
        "learning_rate": neps.Categorical(choices=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "beta1": neps.Categorical(choices=[0.9, 0.95, 0.99]),
        "beta2": neps.Categorical(choices=[0.9, 0.95, 0.99]),
        "weight_decay": neps.Categorical(choices=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        "optimizer_name": neps.Categorical(choices=["adam", "sgd"]),
    }

    root_directory = Path(git.Repo(".", search_parent_directories=True).working_tree_dir) / "output"
    if not root_directory.exists():
        try:
            root_directory.mkdir(parents=True)
        except FileExistsError:
            print("Directory already exists")

    output_tree = (
        f"{args.nodes}_nodes_{args.cpus_per_node}_cpus_"
        f"{args.gpus_per_node}_gpus_{args.n_trainable_layers}_trainable"
    )

    # TODO: set seed for reproducibility across torch and numpy
    neps.run(
        evaluate_pipeline=partial(
            training_pipeline,
            n_trainable_layers=args.n_trainable_layers,
            epochs=20,
        ),
        pipeline_space=pipeline_space,
        optimizer="grid_search",
        root_directory=(f"{root_directory}/{args.group_name}/grid_search/{output_tree}/"),
        max_evaluations_total=500,
        overwrite_working_directory=False,
        post_run_summary=False,
    )
