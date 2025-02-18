import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.resnet import BasicBlock
from tqdm import tqdm


def freeze_layers(
    model: nn.Module, n_unfrozen_layers: int, return_param_stats: bool = False
) -> tuple[int, int, int, list] | None:
    """Freeze all layers except the last n_unfrozen_layers layers."""
    # Freeze all layers
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
    layers_to_unfreeze = (
        layers_with_params[-n_unfrozen_layers:] if n_unfrozen_layers > 0 else []
    )

    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if return_param_stats:
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
    memory: float
    frac_trainable_params: float


class PartitionedResnet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        n_trainable: int = 1,
        compile_backbone: bool = False,
        compile_head: bool = False,
    ) -> None:
        super().__init__()
        base_model = torchvision.models.resnet18(weights=None)
        all_layers: list[tuple] = []
        for child in base_model.children():
            match child:
                case nn.Sequential():
                    for subchild in child.children():
                        all_layers.append((subchild, None))
                case _:
                    all_layers.append((child, None))

        for i, (layer, _) in enumerate(all_layers):
            all_layers[i] = (layer, bool(list(layer.parameters())))

        if n_trainable > sum(1 for _, has_params in all_layers if has_params):
            raise ValueError(
                f"n_trainable is greater than the number of trainable layers: "
                f"{sum(1 for _, has_params in all_layers if has_params)}"
            )

        # create unfrozen head
        ctr = 0
        head_layers = []
        for _, (layer, has_params) in enumerate(all_layers[::-1]):
            if has_params:
                ctr += 1

            if ctr > n_trainable:
                break

            head_layers.append(layer)

        backbone_layers = [layer for layer, _ in all_layers if layer not in head_layers]

        if len(head_layers) > 0:
            head_layers.insert(1, nn.Flatten(1))

        if n_classes != 1000:
            head_layers.remove(head_layers[0])
            head_layers.insert(0, nn.Linear(512, n_classes))

        self.backbone = nn.Sequential(*backbone_layers)
        self.head = nn.Sequential(*head_layers[::-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.head.parameters():
            param.requires_grad = True

        if compile_backbone:
            self.backbone = torch.compile(self.backbone, mode="max-autotune")
        if compile_head:
            self.head = torch.compile(self.head, mode="max-autotune")

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)

        return self.head(x)

    def frac_trainable_params(self) -> float:
        total_params = sum(p.numel() for p in self.backbone.parameters()) + sum(
            p.numel() for p in self.head.parameters()
        )
        frozen_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Total params: {total_params}, Frozen params: {frozen_params}")
        return (total_params - frozen_params) / total_params

    def model_perf_stats(
        self,
        dataloader: DataLoader,
        warmup_passes: int = 100,
        device: torch.device | None = None,
    ) -> ModelPerfStats:
        if device is None:
            device = torch.device("cpu")

        # perform warmup passes
        for i, batch in enumerate(
            tqdm(dataloader, desc="Warming up model", total=warmup_passes)
        ):
            x, _ = batch
            x = x.to(device)
            _ = self(x)
            if i > warmup_passes:
                break

        self.eval()

        fwd_times = []
        bwd_times = []
        for batch in tqdm(dataloader, desc="Measuring forward times"):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            start_time = time.perf_counter()
            pred = self(x)
            fwd_times.append(time.perf_counter() - start_time)

            start_time = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(pred, y)
            loss.backward()
            bwd_times.append(time.perf_counter() - start_time)

        return ModelPerfStats(
            median_fwd_time=np.mean(fwd_times),
            std_fwd_time=np.std(fwd_times),
            median_bwd_time=np.mean(bwd_times),
            std_bwd_time=np.std(bwd_times),
            memory=torch.cuda.memory_allocated(device) / (1024**2),
            frac_trainable_params=self.frac_trainable_params(),
        )


if __name__ == "__main__":
    import torchvision

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    dataset = torch.utils.data.Subset(dataset, range(10000))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
    )

    # compile_settings = [[False, False], [True, True], [True, False], [False, True]]
    compile_settings = [[False, False]]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for compile_model in compile_settings:
        perf_stats = []
        for n_trainable in range(1, 5):
            model = PartitionedResnet(
                n_classes=10,
                n_trainable=n_trainable,
                compile_backbone=compile_model[0],
                compile_head=compile_model[1],
            )
            model = model.to(device)
            stats = model.model_perf_stats(dataloader, device=device)
            # print(model.backbone)
            perf_stats.append(
                (
                    stats.frac_trainable_params,
                    stats.median_fwd_time,
                    stats.std_fwd_time,
                    stats.median_bwd_time,
                    stats.std_bwd_time,
                    stats.memory,
                )
            )

            print(
                f"N trainable: {n_trainable} | Compiled: {compile_model} | "
                f"Median fwd time: {stats.median_fwd_time:.4f}s | Std: {stats.std_fwd_time:.4f}s | "
                f"Median bwd time: {stats.median_bwd_time:.4f}s | Std: {stats.std_bwd_time:.4f}s | "
                f"Memory: {stats.memory:.1f}MB | Trainable params: {stats.frac_trainable_params:.2f}"
            )

        key = (
            f"{'backbone_compiled' if compile_model[0] else 'backbone_not_compiled'}_"
            f"{'head_compiled' if compile_model[1] else 'head_not_compiled'}"
        )
        results[key] = perf_stats

    import matplotlib.pyplot as plt

    # Plot results
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Create second y-axis sharing same x-axis

    for key, stats in results.items():
        frac_trainable, fwd_times, fwd_stds, bwd_times, bwd_stds, memories = zip(
            *stats, strict=False
        )

        # Plot forward times
        ax1.errorbar(
            frac_trainable,
            fwd_times,
            yerr=fwd_stds,
            label=f"{key} (fwd)",
            capsize=3,
            linestyle="-",
        )

        # Plot backward times
        ax1.errorbar(
            frac_trainable,
            bwd_times,
            yerr=bwd_stds,
            label=f"{key} (bwd)",
            capsize=3,
            linestyle="--",
        )

        # Plot memory usage on second y-axis
        ax2.plot(
            frac_trainable, memories, label=f"{key} (memory)", marker="o", linestyle=":"
        )

    ax1.set_xlabel("Number of trainable layers")
    ax1.set_ylabel("Time (s)")
    ax2.set_ylabel("Memory Usage (MB)")

    # Add legends for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Model Performance and Memory Usage")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("model_performance.png")
    plt.close()
