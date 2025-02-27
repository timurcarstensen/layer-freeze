from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class ModelPerfStats:
    median_fwd_time: float
    std_fwd_time: float
    median_bwd_time: float
    std_bwd_time: float
    median_loop_time: float
    memory: float
    frac_trainable_params: float
    optimizer_time: float


def frac_trainable_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(
        p.numel() for p in model.parameters()
    )


def measure_model_training_hw_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    warmup_passes: int = 100,
    device: torch.device | None = None,
) -> ModelPerfStats:
    """Measure the model's forward, backward, and optimizer times."""
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)

    # perform warmup passes
    for i, batch in enumerate(tqdm(dataloader, desc="Warming up model", total=warmup_passes)):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i > warmup_passes:
            break

    fwd_times = []
    bwd_times = []
    loop_times = []
    optimizer_times = []
    max_memory_usage_recorded = 0
    for batch in tqdm(dataloader, desc="Measuring forward times"):
        loop_start = torch.cuda.Event(enable_timing=True)
        loop_end = torch.cuda.Event(enable_timing=True)

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)

        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        opt_start = torch.cuda.Event(enable_timing=True)
        opt_end = torch.cuda.Event(enable_timing=True)

        loop_start.record()
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        fwd_start.record()
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred, y)
        loss = loss.to(device)
        fwd_end.record()
        torch.cuda.synchronize()

        fwd_times.append(fwd_start.elapsed_time(fwd_end))

        bwd_start.record()
        loss.backward()
        bwd_end.record()
        torch.cuda.synchronize()
        bwd_times.append(bwd_start.elapsed_time(bwd_end))

        opt_start.record()
        optimizer.step()
        optimizer.zero_grad()
        opt_end.record()
        torch.cuda.synchronize()
        optimizer_times.append(opt_start.elapsed_time(opt_end))

        loop_end.record()
        torch.cuda.synchronize()
        loop_times.append(loop_start.elapsed_time(loop_end))
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
        frac_trainable_params=frac_trainable_params(model),
        optimizer_time=np.median(optimizer_times),
    )
