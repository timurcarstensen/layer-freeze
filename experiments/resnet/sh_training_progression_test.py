import random
import time
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import models
from tqdm import tqdm

import wandb
from layer_freeze.resnet.utils import data_prep, validate


class FrozenModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_trainable: int | str = 1,
        quantize_frozen_layers: bool = False,
        base_model: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
    ) -> None:
        super().__init__()
        self.n_trainable = n_trainable
        match base_model:
            case "resnet18":
                self.model = models.resnet18(weights=None)
            case "resnet34":
                self.model = models.resnet34(weights=None)
            case "resnet50":
                self.model = models.resnet50(weights=None)
            case "resnet101":
                self.model = models.resnet101(weights=None)
            case "resnet152":
                self.model = models.resnet152(weights=None)
            case _:
                raise ValueError(f"Invalid base model: {base_model}")

        all_layers: list[tuple] = []
        for child in self.model.children():
            match child:
                case nn.Sequential():
                    for subchild in child.children():
                        all_layers.append((subchild, None))
                case _:
                    all_layers.append((child, None))

        for i, (layer, _) in enumerate(all_layers):
            all_layers[i] = (layer, bool(list(layer.parameters())))

        self.n_layers_with_params = sum(1 for _, has_params in all_layers if has_params)

        if isinstance(n_trainable, int):
            if n_trainable > self.n_layers_with_params:
                raise ValueError(
                    f"n_trainable is greater than the number of trainable layers: "
                    f"{self.n_layers_with_params}"
                )

            ctr = 0
            trainable_layers = []
            for _, (layer, has_params) in enumerate(all_layers[::-1]):
                if has_params:
                    ctr += 1

                if ctr > n_trainable:
                    break

                trainable_layers.append(layer)

            frozen_layers = [layer for layer, _ in all_layers if layer not in trainable_layers]

            if len(trainable_layers) > 0:
                trainable_layers.insert(1, nn.Flatten(1))

                if n_classes != 1000:
                    trainable_layers.remove(trainable_layers[0])
                    trainable_layers.insert(0, nn.LazyLinear(n_classes))

            self.frozen_layers = nn.Sequential(*frozen_layers)
            self.trainable_layers = nn.Sequential(*trainable_layers[::-1])

            for param in self.frozen_layers.parameters():
                param.requires_grad = False

            for param in self.trainable_layers.parameters():
                param.requires_grad = True

            self.quantize_frozen_layers = quantize_frozen_layers
            if quantize_frozen_layers:
                self.frozen_layers = self.frozen_layers.to(torch.float16)

    def thaw(self, n_layers: int = 1):
        """takes the last frozen layer from the frozen_layers module and makes it trainable by moving it to the trainable_layers module"""
        frozen_layers = list(self.frozen_layers.children())
        trainable_layers = list(self.trainable_layers.children())
        self.frozen_layers = nn.Sequential(*frozen_layers[:-n_layers])
        self.trainable_layers = nn.Sequential(*frozen_layers[-n_layers:], *trainable_layers)

        if self.quantize_frozen_layers:
            self.frozen_layers = self.frozen_layers.to(torch.float16)
        else:
            self.frozen_layers = self.frozen_layers.to(torch.float32)

        self.trainable_layers = self.trainable_layers.to(torch.float32)

        for param in self.frozen_layers.parameters():
            param.requires_grad = False

        for param in self.trainable_layers.parameters():
            param.requires_grad = True

    def forward(self, x):
        if self.n_trainable == "all":
            return self.model(x)

        if self.quantize_frozen_layers:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = self.frozen_layers(x)
        else:
            x = self.frozen_layers(x)
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            return self.trainable_layers(x)


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    forward_times = []
    backward_times = []
    optimizer_times = []
    for i, (data, target) in enumerate(
        tqdm(trainloader, desc=f"Epoch {epoch}", total=len(trainloader))
    ):
        data = data.to(device)
        target = target.to(device)

        forward_start = time.perf_counter()
        outputs = model(data)
        forward_times.append(time.perf_counter() - forward_start)

        loss = torch.nn.functional.cross_entropy(outputs, target).to(device)

        backward_start = time.perf_counter()
        loss.backward()
        backward_times.append(time.perf_counter() - backward_start)

        start = time.perf_counter()
        optimizer.step()
        optimizer.zero_grad()
        optimizer_times.append(time.perf_counter() - start)

        running_loss += loss.item()
        wandb.log(
            {
                "train/loss": loss.item(),
            }
        )

    wandb.log(
        {
            "train/loss": running_loss / (i + 1),
            "train/forward_time": np.median(forward_times),
            "train/backward_time": np.median(backward_times),
            "train/optimizer_time": np.median(optimizer_times),
            "epoch": epoch,
            "train/frac_trainable": sum(1 for p in model.parameters() if p.requires_grad)
            / sum(1 for p in model.parameters()),
        },
        commit=False,
    )
    return running_loss / (i + 1)


def fidelity_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    schedule: list[tuple[int, int]],
):
    # TODO: increases the fidelity based on the schedule which is list of tuples of (epoch, n_layers)
    # if epoch in schedule:
    #     model.thaw(n_layers=1)

    #     # Get new parameters that require grad but aren't in optimizer
    #     existing_params = {
    #         p for group in optimizer.param_groups for p in group["params"]
    #     }
    ...


def training_pipeline(
    total_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    simulate_sh_training: bool = False,
    train_longer_last_epoch: bool = False,
    last_epoch_multiplier: int = 20,
    quantize_frozen_layers: bool = False,
    fidelity_step_size: int = 1,
) -> dict:
    wandb.init(
        project="sh-training-progression-test",
        entity="timurcarstensen",
        config={
            "total_epochs": total_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "simulate_sh_training": simulate_sh_training,
            "train_longer_last_epoch": train_longer_last_epoch,
            "last_epoch_multiplier": last_epoch_multiplier,
            "quantize_frozen_layers": quantize_frozen_layers,
            "fidelity_step_size": fidelity_step_size,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data
    trainloader, validloader, _, num_classes = data_prep(
        batch_size=batch_size,
        dataloader_workers=16,
        prefetch_factor=4,
    )

    # Define model with new parameters
    model = FrozenModel(
        n_classes=num_classes,
        n_trainable="all" if not simulate_sh_training else 1,
        quantize_frozen_layers=quantize_frozen_layers,
    )
    model = model.to(device)

    # Define loss function and optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=[beta1, beta2],
        weight_decay=weight_decay,
    )

    # Loading potential checkpoint
    epoch = 1
    max_fidelity = model.n_layers_with_params
    epochs_per_fidelity = (total_epochs // max_fidelity) * fidelity_step_size

    if simulate_sh_training:
        print(f"Epochs per fidelity: {epochs_per_fidelity} at step size {fidelity_step_size}")

    while epoch < total_epochs:
        model.train()
        if epoch % epochs_per_fidelity == 0 and simulate_sh_training:
            print(f"Epoch {epoch} thawing")
            model.thaw(n_layers=fidelity_step_size)

            # Get new parameters that require grad but aren't in optimizer
            existing_params = {p for group in optimizer.param_groups for p in group["params"]}
            new_params = [
                p for p in model.parameters() if p.requires_grad and p not in existing_params
            ]

            if new_params:
                # Add new parameters to existing optimizer
                optimizer.add_param_group(
                    {
                        "params": new_params,
                        "lr": learning_rate,
                        "betas": [beta1, beta2],
                        "weight_decay": weight_decay,
                    }
                )

        train(
            model=model,
            trainloader=trainloader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
        )

        with torch.no_grad():
            val_err, val_time = validate(model, validloader, device)

        wandb.log(
            {
                "val/loss": val_err,
                "val/time": val_time,
                "epoch": epoch,
            },
            commit=True,
        )
        epoch += 1

    if train_longer_last_epoch:
        for _ in range(last_epoch_multiplier):
            train(
                model=model,
                trainloader=trainloader,
                device=device,
                optimizer=optimizer,
                epoch=epoch,
            )
            with torch.no_grad():
                val_err, val_time = validate(model, validloader, device)
                wandb.log(
                    {
                        "val/loss": val_err,
                        "val/time": val_time,
                        "epoch": epoch,
                    },
                    commit=True,
                )
            epoch += 1


if __name__ == "__main__":
    config = {
        "beta1": 0.953095018863678,
        "beta2": 0.9431090354919434,
        "learning_rate": 0.0017718230374157429,
        "weight_decay": 1.8971732060890645e-05,
    }

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    training_pipeline(
        total_epochs=100,
        batch_size=256,
        **config,
        simulate_sh_training=True,
        train_longer_last_epoch=True,
        last_epoch_multiplier=100,
        quantize_frozen_layers=False,
        fidelity_step_size=3,
    )
