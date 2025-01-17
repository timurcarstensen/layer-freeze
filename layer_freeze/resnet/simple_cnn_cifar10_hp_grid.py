"""Example adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

import argparse
import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

import git  # noqa: E402
import neps  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from neps.optimizers import GridSearch  # noqa: E402
from neps.plot.tensorboard_eval import tblogger  # noqa: E402
from neps.search_spaces import SearchSpace  # noqa: E402
from tqdm import tqdm  # noqa: E402

from layer_freeze.freeze_layers import freeze_layers  # noqa: E402


def create_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def data_prep(batch_size: int, get_val_set: bool = True) -> tuple:
    """Prepare CIFAR10 dataset for training and testing."""
    # Define dataset specific transforms and classes
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]
    )
    dataset_class = torchvision.datasets.CIFAR10
    num_classes = 10

    trainset = dataset_class(root="./data", train=True, download=True, transform=transform)
    testset = dataset_class(root="./data", train=False, download=True, transform=transform)

    if get_val_set:
        train_size = len(trainset) - 10000  # Reserve 10k samples for validation
        train_set, val_set = torch.utils.data.random_split(trainset, [train_size, 10000])
        validloader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=4
        )
    else:
        train_set = trainset
        validloader = None

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return trainloader, validloader, testloader, num_classes


def full_fidelity_training_pipeline(
    epochs: int = 10,
    batch_size: int = 1024,
    learning_rate: float = 0.008,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    optimizer: str = "adam",
) -> dict:
    """Main training interface for HPO."""
    # Prepare data
    trainloader, validloader, testloader, num_classes = data_prep(batch_size=batch_size)

    # Define model with new parameters
    model = create_model(num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            momentum=beta1,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Loading potential checkpoint
    start_epoch = 1

    if torch.cuda.is_available():
        model = model.cuda()

    # Training loop
    _start = time.time()
    forward_times = []
    backward_times = []
    model.train()
    for epoch in range(start_epoch, epochs + 1):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(
            tqdm(trainloader, desc=f"Epoch {epoch}", leave=False, disable=True), 0
        ):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            forward_start = time.time()
            outputs = model(data)
            forward_end = time.time()
            forward_times.append(forward_end - forward_start)

            loss = criterion(outputs, target)

            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            backward_times.append(backward_end - backward_start)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
        training_loss_for_epoch = running_loss / (batch_idx + 1)
    _end = time.time()

    # Validation loop
    correct = 0
    total = 0
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_err = 1 - (correct / total)

    return {
        "val_err": val_err,
        "val_acc": val_accuracy,
        "cost": _end - _start,
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
    # reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    trainloader, validloader, _, num_classes = data_prep(batch_size=batch_size)

    # Define model with new parameters
    model = create_model(num_classes=num_classes)

    # freeze layers
    freeze_layers(model=model, n_unfrozen_layers=n_unfrozen_layers)

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

    # Loading potential checkpoint
    start_epoch = 1

    model = model.to(device)

    # Training loop
    _start = time.time()
    forward_times = []
    backward_times = []
    model.train()
    for _ in range(start_epoch, epochs + 1):
        running_loss = 0.0
        for i, (data, target) in enumerate(trainloader):  # noqa: B007
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            data = data.to(device)
            target = target.to(device)

            forward_start = time.time()
            outputs = model(data)
            forward_times.append(time.time() - forward_start)

            loss = criterion(outputs, target)

            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_times.append(time.time() - backward_start)

            # print statistics
            running_loss += loss.item()
        training_loss_for_epoch = running_loss / (i + 1)
        # TODO: log training curve
    _end = time.time()

    memory_used = torch.cuda.memory_allocated(device=device) / (1024**2)

    avg_forward_time = sum(forward_times) / len(forward_times)
    avg_backward_time = sum(backward_times) / len(backward_times)

    # Validation loop
    correct = 0
    total = 0
    _val_start = time.time()
    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    val_err = 1 - (correct / total)
    _val_end = time.time()

    # print(f"Accuracy of the network on the 10000 test images: {val_accuracy}")

    # full_fidelity_results = full_fidelity_training_pipeline(
    #     batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     weight_decay=weight_decay,
    #     beta1=beta1,
    #     beta2=beta2,
    #     optimizer=optimizer_name,
    # )
    # Logging
    if log_tensorboard:
        tblogger.log(
            loss=val_err,
            current_epoch=epochs,
            write_summary_incumbent=True,
            writer_config_scalar=True,
            writer_config_hparam=True,
            extra_data={
                "train_loss": tblogger.scalar_logging(loss.item()),
                "val_err": tblogger.scalar_logging(val_err),
                "val_acc": tblogger.scalar_logging(val_accuracy),
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
                "gpu_memory_used_mb": tblogger.scalar_logging(memory_used),
                "avg_forward_time_ms": tblogger.scalar_logging(avg_forward_time * 1000),
                "avg_backward_time_ms": tblogger.scalar_logging(avg_backward_time * 1000),
                # "full_fidelity_val_acc": tblogger.scalar_logging(full_fidelity_results["val_acc"]),
                # "full_fidelity_val_err": tblogger.scalar_logging(full_fidelity_results["val_err"]),
                # "full_fidelity_cost": tblogger.scalar_logging(full_fidelity_results["cost"]),
            },
        )

    return {
        "loss": val_err,
        "cost": _end - _start,
        "info_dict": {
            "train_loss": training_loss_for_epoch,
            "validation_time": _val_end - _val_start,
            "current_epoch": epochs,
            "gpu_memory_used_mb": memory_used,
            "pid": os.getpid(),
            "avg_forward_time_ms": avg_forward_time * 1000,
            "avg_backward_time_ms": avg_backward_time * 1000,
            # "full_fidelity_results": full_fidelity_results,
            "n_trainable_params": sum(
                p.numel() for p in filter(lambda p: p.requires_grad, model.parameters())
            ),
            "n_total_params": sum(p.numel() for p in model.parameters()),
            "perc_trainable_params": (
                sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
                / sum(p.numel() for p in model.parameters())
            )
            * 100,
        },
    }


def get_best_config_id(run_status_path: str) -> int:
    run_status_df = pd.read_csv(run_status_path)
    best_config_id = run_status_df.loc[
        run_status_df["description"] == "best_config_id", "value"
    ].values[0]
    return int(best_config_id)


def get_best_config(config_data_path: str, best_config_id: int) -> pd.DataFrame:
    config_data_df = pd.read_csv(config_data_path)
    best_config_row = config_data_df[config_data_df["config_id"] == best_config_id]
    best_config = {
        "beta1": best_config_row["config.beta1"].values[0],
        "beta2": best_config_row["config.beta2"].values[0],
        "learning_rate": best_config_row["config.learning_rate"].values[0],
        "optimizer": best_config_row["config.optimizer"].values[0],
        "weight_decay": best_config_row["config.weight_decay"].values[0],
    }
    return best_config


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
        f"{args.nodes}nodes_{args.cpus_per_node}cpus_"
        f"{args.gpus_per_node}gpus_{args.n_unfrozen_layers}unfrozen"
    )

    algo = GridSearch

    # TODO: set seed for reproducibility across torch and numpy
    neps.run(
        pipeline_space=pipeline_space,
        run_pipeline=partial(
            training_pipeline,
            log_tensorboard=True,
            n_unfrozen_layers=args.n_unfrozen_layers,
        ),
        searcher=algo(SearchSpace(**pipeline_space), seed=42),
        root_directory=(f"{root_directory}/{args.group_name}/{algo.__name__}/{output_tree}/"),
        max_evaluations_total=500,
        overwrite_working_directory=False,
    )
