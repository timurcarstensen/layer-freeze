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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402

import neps  # noqa: E402
from neps.plot.tensorboard_eval import tblogger  # noqa: E402


def create_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_layers(model: nn.Module, n_unfrozen_layers: int) -> None:
    """Freeze all layers except the last n_unfrozen_layers layers."""
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last n_unfrozen_layers layers
    layers_to_unfreeze = list(model.children())[-n_unfrozen_layers:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True


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
    optimizer: str = "adam",
    log_tensorboard: bool = True,
    dropout_rate: float = 0.0,
    conv_channels: int = 16,
    fc_units: int = 128,
) -> dict:
    """Main training interface for HPO."""
    # Prepare data
    trainloader, validloader, testloader, num_classes = data_prep(batch_size=batch_size)

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
    # if previous_pipeline_directory is not None:
    #     if (Path(previous_pipeline_directory) / "checkpoint.pt").exists():
    #         states = torch.load(
    #             Path(previous_pipeline_directory) / "checkpoint.pt", weights_only=False
    #         )
    #         model = states["model"]
    #         optimizer = states["optimizer"]
    #         start_epoch = states["epochs"]

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
        # print(f'[epoch={epoch},batch={batch_idx+1:<5d}]\tloss: {running_loss / (batch_idx+1):.6f}')
        print(f"epoch={epoch:<2d}\tloss: {running_loss / (batch_idx+1):.6f}")
        training_loss_for_epoch = running_loss / (batch_idx + 1)
    _end = time.time()

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
    _val_end = time.time()

    # TODO: re-enable once neps is fixed
    # Saving checkpoint
    # states = {
    #     "model": model,
    #     "optimizer": optimizer,
    #     "epochs": epochs,
    # }
    # TODO: re-enable once neps is fixed
    # torch.save(states, Path(pipeline_directory) / "checkpoint.pt")

    print(f"Accuracy of the network on the 10000 test images: {val_accuracy}")

    # Logging
    if log_tensorboard:
        tblogger.log(
            loss=val_err,
            current_epoch=epochs,
            # Set to `True` for a live incumbent trajectory.
            write_summary_incumbent=True,
            # Set to `True` for a live loss trajectory for each config.
            writer_config_scalar=True,
            # Set to `True` for live parallel coordinate, scatter plot matrix, and table view.
            writer_config_hparam=True,
            # Appending extra data
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
                "gpu_memory_used_mb": tblogger.scalar_logging(
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                ),
                "avg_forward_time_ms": tblogger.scalar_logging(avg_forward_time * 1000),
                "avg_backward_time_ms": tblogger.scalar_logging(avg_backward_time * 1000),
            },
        )

    return {
        "loss": val_err,
        "cost": _end - _start,
        "info_dict": {
            "train_loss": training_loss_for_epoch,
            "validation_time": _val_end - _val_start,
            "current_epoch": epochs,
            "gpu_memory_used_mb": torch.cuda.max_memory_allocated()
            / (1024 * 1024),  # Convert to MB
            "pid": os.getpid(),
            "avg_forward_time_ms": avg_forward_time * 1000,
            "avg_backward_time_ms": avg_backward_time * 1000,
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
        "dropout_rate": best_config_row["config.dropout_rate"].values[0],
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
        "learning_rate": neps.Float(1e-5, 1e-1, log=True, default=0.008),
        "beta1": neps.Float(0.9, 0.999, log=True, default=0.9),
        "beta2": neps.Float(0.9, 0.999, log=True, default=0.999),
        "weight_decay": neps.Float(1e-5, 0.1, log=True, default=0.01),
        "optimizer": neps.Categorical(["adam", "sgd"], default="adam"),
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
    neps.run(
        pipeline_space=pipeline_space,
        run_pipeline=partial(
            training_pipeline,
            log_tensorboard=True,
            n_unfrozen_layers=args.n_unfrozen_layers,
        ),
        searcher=algo,
        max_evaluations_total=2,
        root_directory=(f"{root_directory}/" f"{args.group_name}/{algo}/{output_tree}/"),
        overwrite_working_directory=False,
    )

    config_files_dir = root_directory / output_tree / "summary_csv"
    best_config_id = get_best_config_id(config_files_dir / "run_status.csv")
    best_config = get_best_config(config_files_dir / "config_data.csv", best_config_id)
    print(best_config)

    # Evaluate best config with all layers unfrozen
    print("\nEvaluating best config with all layers unfrozen...")
    best_config_dict = best_config.to_dict()
    evaluation_result = training_pipeline(
        log_tensorboard=True,
        **best_config_dict,
        n_unfrozen_layers=num_layers,  # Use all layers
    )
    print(f"\nFull model evaluation result: {evaluation_result}")
