from copy import deepcopy

import torch.nn as nn
import torchao
from torchvision.models.resnet import BasicBlock


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
                all_layers.append((layer, hasattr(layer, "weight")))

    get_all_layers(module=model)

    layers_with_params = [layer for layer, has_weight in all_layers if has_weight]
    layers_to_unfreeze = layers_with_params[-n_unfrozen_layers:] if n_unfrozen_layers > 0 else []

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


class CustomPartitionedModel(nn.Module):
    def __init__(
        self, n_trainable: int = 1, jit_freeze: bool = False, quantize_backbone: bool = False
    ) -> None:
        super().__init__()
        base_model = torchvision.models.resnet18()
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

        # create unfrozen head
        ctr = 0
        head_layers = []
        for i, (layer, has_params) in enumerate(all_layers[::-1]):
            if has_params:
                ctr += 1

            if ctr > n_trainable:
                break

            head_layers.append(deepcopy(layer))
            print(layer)

        backbone_layers = [layer for layer, _ in all_layers if layer not in head_layers]

        self.backbone = nn.Sequential(*backbone_layers)
        self.head = nn.Sequential(*head_layers[::-1])

        if quantize_backbone:
            # torch.backends.quantized.engine = "qnnpack"
            # quantized_layers = [
            #     convert(
            #         layer,
            #         {
            #             "nn.Linear": torch.ao.nn.quantized.Linear,
            #             "nn.Conv2d": torch.ao.nn.quantized.Conv2d,
            #             "nn.BatchNorm2d": torch.ao.nn.quantized.BatchNorm2d,
            #         },
            #     )
            #     for layer in self.backbone.children()
            # ]
            # backbone_layers = [layer for layer in self.backbone.children()]
            self.backbone = torchao.autoquant(self.backbone)

            # pdb.set_trace()
            # self.backbone = nn.Sequential(
            #     *
            # )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


if __name__ == "__main__":
    import torchvision
    # from torch.jit import freeze

    # mod = torchvision.models.resnet18()

    # frozen_params, total_params, layers_with_params, layers = freeze_layers(
    #     model=mod, n_unfrozen_layers=2, return_param_stats=True
    # )
    # import pdb

    # pdb.set_trace()
    # print(mod)
    # print(layers_with_params)
    # print(layers)

    model = CustomPartitionedModel(2, quantize_backbone=True)
    print(model)
