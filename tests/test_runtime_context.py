import importlib
import sys
import types

import torch


def _fake_wan_modules(monkeypatch):
    wan_module = types.ModuleType("wan")
    modules_module = types.ModuleType("wan.modules")
    model_module = types.ModuleType("wan.modules.model")
    wan_module.modules = modules_module
    modules_module.model = model_module

    class WanSelfAttention:
        def __init__(self):
            pass

        def forward(self, x, seq_lens, grid_sizes, freqs):
            return x

    class WanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, x, t):
            return x * self.weight

    model_module.WanSelfAttention = WanSelfAttention
    model_module.WanModel = WanModel
    monkeypatch.setattr(sys, "modules", {
        **sys.modules,
        "wan": wan_module,
        "wan.modules": modules_module,
        "wan.modules.model": model_module,
    })
    return model_module


def test_runtime_context_injects_step_branch_layer_and_grid(monkeypatch):
    model_module = _fake_wan_modules(monkeypatch)
    context = importlib.import_module("attention_backends.context")
    context.install_wan_context()

    model = model_module.WanModel()
    self_attention = model_module.WanSelfAttention()
    model(torch.ones(()), torch.tensor([10.0]))
    self_attention.forward(
        torch.ones(()), None, torch.tensor([[2, 3, 4]]), None)
    assert context.routing_context(include_grid=True) == (0, 0, 0, (2, 3, 4))

    model(torch.ones(()), torch.tensor([10.0]))
    assert context.routing_context()[1:] == (0, 1)
    model(torch.ones(()), torch.tensor([9.0]))
    assert context.routing_context()[1:] == (1, 0)

    profile = context.runtime_profile()
    assert profile["denoising_steps_observed"] == 2
    assert profile["model_forward_calls"] == 3


def test_flex_config_static_default_and_validation():
    from attention_backends.flex import FlexReuseBackend, FlexReuseConfig

    assert FlexReuseConfig().update_interval == 0
    for config in (
        FlexReuseConfig(update_interval=-1),
        FlexReuseConfig(keep=0),
        FlexReuseConfig(mass_target=1.1),
        FlexReuseConfig(update_interval=1, sampled_update_interval=1),
        FlexReuseConfig(route_samples=129),
        FlexReuseConfig(route_persistence=-0.1),
        FlexReuseConfig(bootstrap_mode="invalid"),
        FlexReuseConfig(dense_route_threshold=0),
        FlexReuseConfig(directional_update=True, spatial_reorder=False),
        FlexReuseConfig(route_budget_scale=0),
    ):
        try:
            FlexReuseBackend(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config was accepted: {config}")
