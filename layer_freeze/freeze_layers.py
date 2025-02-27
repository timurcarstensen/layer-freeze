import csv
import gc
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm


def freeze_layers(model: nn.Module, n_trainable_layers: int) -> tuple[int, int, int, list]:
    """Freeze all layers except the last n_unfrozen_layers layers."""
    # Freeze all layers
    # TODO: make it so that this skips over layers that don't have weights
    for param in model.parameters():
        param.requires_grad = False

    # Get all layers including nested ones
    all_layers = []

    def get_all_layers(module: nn.Module) -> None:
        for layer in module.children():
            if isinstance(layer, (nn.Sequential, BasicBlock)):
                get_all_layers(layer)
            else:
                all_layers.append((layer, bool(list(layer.parameters()))))

    get_all_layers(module=model)

    layers_with_params = [layer for layer, has_weight in all_layers if has_weight]

    if n_trainable_layers > len(layers_with_params):
        raise ValueError(
            f"n_trainable_layers is greater than the number of layers with parameters: {len(layers_with_params)}"
        )

    layers_to_unfreeze = layers_with_params[-n_trainable_layers:] if n_trainable_layers > 0 else []

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # if return_param_stats:
    return (
        frozen_params,
        total_params,
        sum(1 for _, has_weight in all_layers if has_weight),
        [layer for layer, has_weight in all_layers if has_weight],
    )


@dataclass
class ModelPerfStats:
    median_fwd_time: float
    std_fwd_time: float
    median_bwd_time: float
    std_bwd_time: float
    median_loop_time: float
    memory: float
    frac_trainable_params: float


class FrozenModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_trainable: int,
        base_model: Literal[
            "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        ] = "resnet18",
        compile: bool = False,
    ) -> None:
        super().__init__()
        match base_model:
            case "resnet18":
                self.model = torchvision.models.resnet18(weights=None)
            case "resnet34":
                self.model = torchvision.models.resnet34(weights=None)
            case "resnet50":
                self.model = torchvision.models.resnet50(weights=None)
            case "resnet101":
                self.model = torchvision.models.resnet101(weights=None)
            case "resnet152":
                self.model = torchvision.models.resnet152(weights=None)
            case _:
                raise ValueError(f"Invalid base model: {base_model}")

        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

        frozen_params, total_params, n_layers, layers = freeze_layers(
            self.model, n_trainable_layers=n_trainable
        )
        self.n_frozen_params = frozen_params
        self.n_total_params = total_params
        self.n_layers_with_params = n_layers
        self.layers = layers

        # if compile:
        #     self.model = torch.compile(self.model, mode="reduce-overhead")

    # def __init__(
    #     self,
    #     n_classes: int,
    #     n_trainable: int = 1,
    #     compile_backbone: bool = False,
    #     compile_head: bool = False,
    # ) -> None:
    #     super().__init__()
    #     base_model = torchvision.models.resnet18(weights=None)
    #     all_layers: list[tuple] = []
    #     for child in base_model.children():
    #         match child:
    #             case nn.Sequential():
    #                 for subchild in child.children():
    #                     all_layers.append((subchild, None))
    #             case _:
    #                 all_layers.append((child, None))

    #     for i, (layer, _) in enumerate(all_layers):
    #         all_layers[i] = (layer, bool(list(layer.parameters())))

    #     if n_trainable > sum(1 for _, has_params in all_layers if has_params):
    #         raise ValueError(
    #             f"n_trainable is greater than the number of trainable layers: "
    #             f"{sum(1 for _, has_params in all_layers if has_params)}"
    #         )

    #     # create unfrozen head
    #     ctr = 0
    #     head_layers = []
    #     for _, (layer, has_params) in enumerate(all_layers[::-1]):
    #         if has_params:
    #             ctr += 1

    #         if ctr > n_trainable:
    #             break

    #         head_layers.append(layer)

    #     backbone_layers = [layer for layer, _ in all_layers if layer not in head_layers]

    #     if len(head_layers) > 0:
    #         head_layers.insert(1, nn.Flatten(1))

    #     if n_classes != 1000:
    #         head_layers.remove(head_layers[0])
    #         head_layers.insert(0, nn.Linear(512, n_classes))

    #     self.backbone = nn.Sequential(*backbone_layers)
    #     self.head = nn.Sequential(*head_layers[::-1])

    #     for param in self.backbone.parameters():
    #         param.requires_grad = False

    #     for param in self.head.parameters():
    #         param.requires_grad = True

    # if compile_backbone:
    #     self.backbone = torch.compile(self.backbone, mode="max-autotune")
    # if compile_head:
    #     self.head = torch.compile(self.head, mode="max-autotune")

    def forward(self, x):
        return self.model(x)

    def frac_trainable_params(self) -> float:
        return (self.n_total_params - self.n_frozen_params) / self.n_total_params

    def model_perf_stats(
        self,
        dataloader: "DataLoader",
        optimizer: torch.optim.Optimizer,
        warmup_passes: int = 100,
        device: torch.device | None = None,
    ) -> ModelPerfStats:
        if device is None:
            device = torch.device("cpu")

        self = self.to(device)

        # perform warmup passes
        for i, batch in enumerate(tqdm(dataloader, desc="Warming up model", total=warmup_passes)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            pred = self(x)
            loss = torch.nn.functional.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i > warmup_passes:
                break

        fwd_times = []
        bwd_times = []
        loop_times = []
        max_memory_usage_recorded = 0
        for batch in tqdm(dataloader, desc="Measuring forward times"):
            loop_start_time = time.perf_counter()
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            start_time = time.perf_counter()
            pred = self(x)
            fwd_times.append(time.perf_counter() - start_time)

            loss = torch.nn.functional.cross_entropy(pred, y)
            start_time = time.perf_counter()
            loss = loss.to(device)
            loss.backward()
            bwd_times.append(time.perf_counter() - start_time)
            optimizer.step()
            optimizer.zero_grad()
            loop_times.append(time.perf_counter() - loop_start_time)
            max_memory_usage_recorded = max(
                max_memory_usage_recorded, torch.cuda.memory_allocated(device) / (1024**2)
            )

        assert torch.cuda.memory_allocated(device) / (1024**2) == max_memory_usage_recorded
        return ModelPerfStats(
            median_fwd_time=np.median(fwd_times),
            std_fwd_time=np.std(fwd_times),
            median_bwd_time=np.median(bwd_times),
            std_bwd_time=np.std(bwd_times),
            median_loop_time=np.median(loop_times),
            memory=max_memory_usage_recorded,
            frac_trainable_params=self.frac_trainable_params(),
        )


if __name__ == "__main__":
    import argparse
    import logging
    import sys

    import matplotlib.pyplot as plt
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--measure_iterations", type=int, default=100)
    parser.add_argument("--warmup_passes", type=int, default=30)
    args = parser.parse_args()

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset = torch.utils.data.Subset(dataset, range(args.measure_iterations * args.batch_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
    )

    results: list[ModelPerfStats] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n_trainable in range(
        1,
        FrozenModel(n_classes=10, n_trainable=1, base_model=args.base_model).n_layers_with_params,
    ):
        model = FrozenModel(
            n_classes=10,
            n_trainable=n_trainable,
            base_model=args.base_model,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        stats = model.model_perf_stats(
            dataloader, optimizer=optimizer, device=device, warmup_passes=args.warmup_passes
        )

        root.info(
            f"N trainable: {n_trainable} | "
            f"Median fwd time: {stats.median_fwd_time:.4f}s | Std: {stats.std_fwd_time:.4f}s | "
            f"Median bwd time: {stats.median_bwd_time:.4f}s | Std: {stats.std_bwd_time:.4f}s | "
            f"Median loop time: {stats.median_loop_time:.4f}s | "
            f"Memory: {stats.memory:.1f}MB | Trainable params: {stats.frac_trainable_params:.4f}"
        )

        results.append(stats)
        del model
        del optimizer
        del stats
        torch.cuda.empty_cache()
        gc.collect()
        root.info(
            "memory usage after empty cache is %s", torch.cuda.memory_allocated(device) / (1024**2)
        )

    # Plot results
    fig, ax1 = plt.subplots(figsize=(7, 7))
    ax2 = ax1.twinx()  # Create second y-axis sharing same x-axis

    fwd_times = [s.median_fwd_time for s in results]
    fwd_stds = [s.std_fwd_time for s in results]
    bwd_times = [s.median_bwd_time for s in results]
    bwd_stds = [s.std_bwd_time for s in results]
    memories = [s.memory for s in results]
    loop_times = [s.median_loop_time for s in results]
    frac_trainable = [s.frac_trainable_params for s in results]

    # Save results to CSV
    csv_filename = (
        f"./results/model_perf/data/model_stats_{args.base_model}_batch_size_{args.batch_size}.csv"
    )
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frac_trainable",
                "fwd_time",
                "fwd_std",
                "bwd_time",
                "bwd_std",
                "memory",
                "median_loop_time",
            ]
        )
        for i in range(len(results)):
            writer.writerow(
                [
                    frac_trainable[i],
                    fwd_times[i],
                    fwd_stds[i],
                    bwd_times[i],
                    bwd_stds[i],
                    memories[i],
                    loop_times[i],
                ]
            )

    # Plot forward times
    ax1.plot(
        frac_trainable,
        fwd_times,
        linestyle="-",
        label="forward",
    )

    # Plot backward times
    ax1.plot(
        frac_trainable,
        bwd_times,
        linestyle="--",
        label="backward",
    )

    # Plot memory usage on second y-axis
    ax2.plot(
        frac_trainable,
        memories,
        label="memory",
        marker="o",
        linestyle=":",
    )

    ax1.set_xlabel("Fraction of trainable params")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Memory Usage (MB)")

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3
    )

    # ax1.grid(True, axis="both", linestyle="--", alpha=0.4, which="major")
    # ax2.grid(True, axis="both", linestyle="--", alpha=0.4, which="major")

    plt.title(
        f"{args.base_model} Model Performance and Memory Usage with batch size {args.batch_size}"
    )
    plt.grid(True, axis="both", linestyle="--", alpha=0.4, which="major")
    plt.tight_layout()
    plt.savefig(
        f"./results/model_perf/plots/model_performance_{args.base_model}_batch_size_{args.batch_size}.pdf",
        dpi=500,
    )
    plt.close()
