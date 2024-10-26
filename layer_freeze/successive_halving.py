import math
from typing import Any, Dict, List

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm, trange


def create_model(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_layers(model: nn.Module, num_unfrozen: int) -> None:
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last num_unfrozen layers
    layers_to_unfreeze = list(model.children())[-num_unfrozen:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    num_unfrozen_layers: int,
) -> float:
    # Hyperparameters to optimize
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Create model and freeze layers
    model = create_model()
    freeze_layers(model, num_unfrozen_layers)

    # Create optimizer
    optimizer: optim.Optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9
        )

    # Training loop
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for _ in trange(epochs, desc="Epochs"):
        model.train()
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = correct / total
    return accuracy


def successive_halving(
    n_trials: int = 27, eta: int = 3, min_layers: int = 1, max_layers: int = 8, epochs: int = 10
) -> Dict[str, Any]:
    # Load and preprocess data
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    batch_size = 128  # Fixed batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Calculate the number of brackets
    s_max = math.floor(math.log(max_layers / min_layers, eta))
    B = (s_max + 1) * n_trials

    # Create a list to store all trials
    all_trials: List[optuna.Trial] = []

    # Successive Halving main loop
    for s in trange(s_max, -1, -1, desc="Brackets"):
        # Number of configurations for this bracket
        n = math.ceil(B / n_trials / (s + 1) * eta**s)
        # Number of unfrozen layers for this bracket
        num_unfrozen = min(max_layers, math.ceil(max_layers * eta ** (-s)))

        # Create a study object for this bracket
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())

        # Evaluate each configuration
        for _ in trange(n, desc="Configurations"):
            trial = study.ask()
            value = objective(trial, train_loader, val_loader, epochs, num_unfrozen)
            study.tell(trial, value)
            all_trials.append(trial)

        # Promote the top 1/eta configurations to the next round
        top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[: math.ceil(n / eta)]

    # Find the best trial among all brackets
    best_trial = max(all_trials, key=lambda t: t.value)

    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial.params


if __name__ == "__main__":
    best_params = successive_halving()
    print(f"Best hyperparameters: {best_params}")
