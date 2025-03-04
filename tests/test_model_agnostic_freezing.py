import pytest
import torch
import torch.nn as nn
from torchvision.models import ResNet, VisionTransformer, resnet18, vit_b_16

from layer_freeze.model_agnostic_freezing import FrozenModel


@pytest.fixture
def resnet_model():
    return resnet18(weights=None)


@pytest.fixture
def vit_model():
    return vit_b_16(weights=None)


def test_resnet_basic_freezing(resnet_model):
    """Test basic freezing behavior with ResNet."""
    frozen_model = FrozenModel(resnet_model, n_trainable=1, unwrap=ResNet)

    # Last layer (fc) should be trainable
    assert all(p.requires_grad for p in frozen_model.trainable[0].parameters())
    # All other layers should be frozen
    assert all(not p.requires_grad for p in frozen_model.frozen[0].parameters())


def test_vit_basic_freezing(vit_model):
    """Test basic freezing behavior with ViT."""
    frozen_model = FrozenModel(vit_model, n_trainable=1, unwrap=VisionTransformer)

    # Check that only the last layer is trainable
    assert all(p.requires_grad for p in frozen_model.trainable[0].parameters())
    # All other layers should be frozen
    assert all(not p.requires_grad for p in frozen_model.frozen[0].parameters())


def test_resnet_all_trainable(resnet_model):
    """Test setting all layers as trainable."""
    max_fid = FrozenModel(resnet_model, n_trainable=1, unwrap=ResNet).max_fidelity
    model = FrozenModel(resnet_model, n_trainable=max_fid, unwrap=ResNet)

    # All parameters should be trainable
    assert all(p.requires_grad for p in model.parameters())


def test_vit_all_trainable(vit_model):
    """Test setting all layers as trainable."""
    max_fid = FrozenModel(vit_model, n_trainable=1, unwrap=VisionTransformer).max_fidelity
    model = FrozenModel(vit_model, n_trainable=max_fid, unwrap=VisionTransformer)

    # All parameters should be trainable
    assert all(p.requires_grad for p in model.parameters())


def test_resnet_thaw(resnet_model):
    """Test thawing additional layers."""
    frozen_model = FrozenModel(resnet_model, n_trainable=1, unwrap=ResNet)
    initial_trainable = len(frozen_model.trainable)

    frozen_model.thaw(1)
    assert len(frozen_model.trainable) > initial_trainable


def test_vit_thaw(vit_model):
    """Test thawing additional layers."""
    frozen_model = FrozenModel(vit_model, n_trainable=1, unwrap=VisionTransformer)
    initial_trainable = len(frozen_model.trainable)

    frozen_model.thaw(1)
    assert len(frozen_model.trainable) > initial_trainable


def test_resnet_forward_pass(resnet_model):
    """Test that forward pass works correctly."""
    frozen_model = FrozenModel(resnet_model, n_trainable=1, unwrap=ResNet)
    x = torch.randn(1, 3, 224, 224)

    # Should work without errors
    output = frozen_model(x)
    assert output.shape == (1, 1000)  # ResNet18 output shape


def test_vit_forward_pass(vit_model):
    """Test that forward pass works correctly."""
    frozen_model = FrozenModel(vit_model, n_trainable=1, unwrap=VisionTransformer)
    x = torch.randn(1, 3, 224, 224)

    # Should work without errors
    output = frozen_model(x)
    assert output.shape == (1, 1000)  # ViT output shape


def test_invalid_n_trainable(resnet_model):
    """Test that invalid n_trainable raises ValueError."""
    with pytest.raises(ValueError):
        FrozenModel(resnet_model, n_trainable=1000, unwrap=ResNet)  # Too many trainable layers


def test_resnet_parameter_counts(resnet_model):
    """Test that parameter counts are preserved."""
    original_params = sum(p.numel() for p in resnet_model.parameters())
    frozen_model = FrozenModel(resnet_model, n_trainable=1, unwrap=ResNet)

    frozen_params = sum(p.numel() for p in nn.Sequential(*frozen_model.frozen).parameters())
    trainable_params = sum(p.numel() for p in nn.Sequential(*frozen_model.trainable).parameters())

    assert frozen_params + trainable_params == original_params
