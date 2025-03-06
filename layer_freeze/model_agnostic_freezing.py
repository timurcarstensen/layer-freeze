from typing import Any

import rich
import torch.nn as nn


class FrozenModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        n_trainable: int = 1,
        unwrap: Any | list[Any] = None,
        print_summary: bool = False,
    ) -> None:
        """Initialize a FrozenModel that wraps a base model and freezes some of its layers.

        Args:
            base_model (nn.Module): The PyTorch model to wrap and partially freeze
            n_trainable (int | str, optional): Number of layers to keep trainable, counting from the end.
                Defaults to 1.
            unwrap (Any, optional): A class/type to unwrap during layer traversal. If a layer matches
                this type, its children will be traversed instead of treating it as a leaf node.
                Defaults to None.
            print_summary (bool, optional): Whether to print a summary of frozen/trainable layers
                and parameter counts after initialization. Defaults to False.

        Raises:
            ValueError: If n_trainable is greater than the number of layers with parameters
        """
        super().__init__()
        self.n_trainable = n_trainable
        self.base_model = base_model

        # unwrap = nn.Sequential if unwrap is None else unwrap
        all_layers: list[tuple] = []

        unwrap = nn.Sequential if unwrap is None else unwrap

        def _recursive_children(mod: nn.Module, unwrap: Any = None) -> None:
            if isinstance(mod, (nn.Sequential, unwrap)):
                for child in mod.children():
                    _recursive_children(child, unwrap)

            else:
                all_layers.append((mod, None))

        _recursive_children(self.base_model, unwrap)

        for i, (layer, _) in enumerate(all_layers):
            all_layers[i] = (layer, bool(list(layer.parameters())))

        self.all_layers = all_layers  # Store as instance variable

        n_layers_with_params = sum(1 for _, has_params in self.all_layers if has_params)

        self.max_fidelity = n_layers_with_params

        if n_trainable > n_layers_with_params:
            raise ValueError(
                f"n_trainable is greater than the number of trainable layers: "
                f"{n_layers_with_params}"
            )

        self.frozen, self.trainable = self._split_layers_and_freeze(self.all_layers, n_trainable)

        assert len(self.frozen) + len(self.trainable) == len(self.all_layers)

        if print_summary:
            self.print_layers()

    @staticmethod
    def _split_layers_and_freeze(
        layers: list[tuple[nn.Module, bool]], n_trainable: int
    ) -> tuple[list[nn.Module], list[nn.Module]]:
        ctr = 0
        trainable: list[nn.Module] = []
        for _, (layer, has_params) in enumerate(layers[::-1]):
            if has_params:
                ctr += 1

            if ctr <= n_trainable:
                trainable.insert(0, layer)
            else:
                break

        frozen: list[nn.Module] = [layer for layer, _ in layers if layer not in trainable]

        for layer in frozen:
            for param in layer.parameters():
                param.requires_grad = False

        for layer in trainable:
            for param in layer.parameters():
                param.requires_grad = True

        return frozen, trainable

    def thaw(self, n: int = 1):
        """
        makes the last `n` frozen layers trainable.
        """
        self.n_trainable += n
        if not self.n_trainable > self.max_fidelity:
            self.frozen, self.trainable = self._split_layers_and_freeze(
                self.all_layers, self.n_trainable
            )
        else:
            raise ValueError("Model already fully thawed.")

    def forward(self, x):
        return self.base_model(x)

    def print_layers(self):
        """
        Prints all layers showing their trainable status and parameter count,
        grouped by frozen/trainable status.
        """
        _frozen = []
        _trainable = []

        # Group layers by their trainable status
        for layer in self.frozen:
            # Count parameters and check if any are trainable
            num_params = sum(p.numel() for p in layer.parameters())
            # is_trainable = any(p.requires_grad for p in layer.parameters())

            layer_info = {
                "name": layer.__class__.__name__,
                "num_params": num_params,
                "trainable": all(not p.requires_grad for p in layer.parameters()),
            }
            if layer_info["num_params"] > 0:
                _frozen.append(layer_info)

        for layer in self.trainable:
            num_params = sum(p.numel() for p in layer.parameters())
            # is_trainable = any(p.requires_grad for p in layer.parameters())

            layer_info = {
                "name": layer.__class__.__name__,
                "num_params": num_params,
                "trainable": all(p.requires_grad for p in layer.parameters()),
            }
            if layer_info["num_params"] > 0:
                _trainable.append(layer_info)

        rich.print("\n[bold blue]Frozen Layers:[/bold blue]")
        for layer in _frozen:
            rich.print(f"  {layer['name']}, trainable = False, params = {layer['num_params']:,}")

        if len(_trainable) > 0:
            rich.print("\n[bold green]Trainable Layers:[/bold green]")
            for layer in _trainable:
                rich.print(f"  {layer['name']}, trainable = True, params = {layer['num_params']:,}")
        else:
            rich.print("[bold red]No Trainable Layers[/bold red]")

        total_frozen = sum(layer["num_params"] for layer in _frozen)
        total_trainable = sum(layer["num_params"] for layer in _trainable)
        total_base_model = sum(p.numel() for p in self.base_model.parameters())
        rich.print("[bold]Summary:[/bold]")
        rich.print(f"  Total frozen parameters: {total_frozen:,}")
        rich.print(f"  Total trainable parameters: {total_trainable:,}")
        rich.print(f"  Total parameters: {total_frozen + total_trainable:,}")
        rich.print(f"  Total base model parameters: {total_base_model:,}")
