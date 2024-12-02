"""Example adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html"""

import argparse
import logging
import os
import time
import warnings
from functools import partial
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import git  # noqa: E402
import neps  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
import torchvision  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
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
            val_set, batch_size=batch_size, shuffle=False, num_workers=1
        )
    else:
        train_set = trainset
        validloader = None

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return trainloader, validloader, testloader, num_classes


def training_pipeline(
    pipeline_directory: str,
    previous_pipeline_directory: str | None,
    epochs: int = 10,
    n_unfrozen_layers: int = 1,
    batch_size: int = 64,
    learning_rate: float = 0.001,
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

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=[beta1, beta2],
            weight_decay=weight_decay,
        )
    elif optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
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
    for epoch in range(start_epoch, epochs + 1):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        # print(f'[epoch={epoch},batch={batch_idx+1:<5d}]\tloss: {running_loss / (batch_idx+1):.6f}')
        print(f"epoch={epoch:<2d}\tloss: {running_loss / (batch_idx+1):.6f}")
        training_loss_for_epoch = running_loss / (batch_idx + 1)
    _end = time.time()

    # Validation loop
    correct = 0
    total = 0
    _val_start = time.time()
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
            },
        )

    return {
        "loss": val_err,
        "cost": _end - _start,
        "info_dict": {
            "train_loss": training_loss_for_epoch,
            "validation_time": _val_end - _val_start,
            "current_epoch": epochs,
            "pid": os.getpid(),
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
    model = Net()
    num_layers = len(list(model.children()))
    if args.n_unfrozen_layers > num_layers:
        raise ValueError(
            f"n_unfrozen_layers ({args.n_unfrozen_layers}) must be <= {num_layers}, "
            f"the total number of layers in Net"
        )

    pipeline_space = {
        "learning_rate": neps.Float(1e-5, 1e-1, log=True, default=0.001),
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
        ),
        searcher=algo,
        max_evaluations_total=500,
        root_directory=(f"{root_directory}/" f"{args.group_name}/{algo}/{output_tree}/"),
        overwrite_working_directory=False,
    )
