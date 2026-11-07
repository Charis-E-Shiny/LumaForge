import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lumaforge.hybrid import prepare
from lumaforge.metrics import isp_loss
from lumaforge.models.lumanet import LumaNet, NetConfig


def tensors(size=64):
    return torch.rand(2, 4, size // 2, size // 2), torch.rand(2, 1, size // 2, size // 2), torch.rand(2, 3, size, size)


def test_hybrid_starts_as_identity_of_base():
    packed, noise, base = tensors()
    out = LumaNet(NetConfig(width=8)).eval()(packed, noise, base)
    assert torch.allclose(out, base, atol=1e-6)


@pytest.mark.parametrize(
    "config",
    [NetConfig(width=8, hybrid=False), NetConfig(width=8, use_noise_map=False), NetConfig(width=8, use_color_head=False)],
)
def test_variants_produce_full_resolution(config):
    packed, noise, base = tensors()
    assert LumaNet(config)(packed, noise, base).shape == base.shape


def test_loss_is_differentiable_and_decreases():
    packed, noise, base = tensors()
    target = (base * 0.8 + 0.1).detach()
    model = LumaNet(NetConfig(width=8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        loss = isp_loss(model(packed, noise, base), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_prepare_crops_to_network_stride():
    bayer = np.random.default_rng(0).random((202, 250)).astype(np.float32) * 0.5
    inputs = prepare(bayer, "IP")
    assert inputs.base.shape == (3, 192, 240)
    assert inputs.packed.shape == (4, 96, 120)
    assert inputs.noise.shape == (1, 96, 120)
