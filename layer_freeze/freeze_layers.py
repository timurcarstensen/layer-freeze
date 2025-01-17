import torch.nn as nn


def freeze_layers(
    model: nn.Module, n_unfrozen_layers: int, return_param_stats: bool = False
) -> tuple[int, int]:
    """Freeze all layers except the last n_unfrozen_layers layers."""
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Get all layers including nested ones
    all_layers = []

    def get_all_layers(module: nn.Module) -> None:
        for layer in module.children():
            if isinstance(layer, nn.Sequential) or "torchvision" in layer.__module__:
                get_all_layers(layer)
            else:
                all_layers.append(layer)

    get_all_layers(model)
    layers_to_unfreeze = all_layers[-n_unfrozen_layers:] if n_unfrozen_layers > 0 else []
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    if return_param_stats:
        return frozen_params, total_params
