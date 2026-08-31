import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_install_probe_patches_both_attention_call_sites(monkeypatch):
    wan_module = types.ModuleType("wan")
    modules_module = types.ModuleType("wan.modules")
    attention_module = types.ModuleType("wan.modules.attention")
    model_module = types.ModuleType("wan.modules.model")
    wan_module.modules = modules_module
    modules_module.attention = attention_module
    modules_module.model = model_module

    def original_attention(*args, **kwargs):
        return "dense"

    def original_forward(instance, *args, **kwargs):
        return "forward"

    class FakeSelfAttention:
        def __init__(self, *args, **kwargs):
            pass

        def forward(self, *args, **kwargs):
            return "self"

    attention_module.flash_attention = original_attention
    model_module.flash_attention = original_attention
    model_module.WanSelfAttention = FakeSelfAttention
    model_module.WanModel = types.SimpleNamespace(forward=original_forward)

    modules = {
        "wan": wan_module,
        "wan.modules": modules_module,
        "wan.modules.attention": attention_module,
        "wan.modules.model": model_module,
    }
    monkeypatch.setattr(sys, "modules", {**sys.modules, **modules})

    runner = importlib.import_module("profiles._runner")

    timesteps = []
    calls = []

    def attention_hook(original, model, args, kwargs):
        calls.append(original)
        assert model is model_module
        return original(*args, **kwargs)

    runner.install_probe(attention_hook, timesteps.append)

    assert attention_module.flash_attention is not original_attention
    assert model_module.flash_attention is attention_module.flash_attention
    assert attention_module.flash_attention() == "dense"
    assert calls == [original_attention]
    assert model_module.WanModel.forward(None) == "forward"

    model_module.WanModel.forward(None, t=3)
    assert timesteps == [3]
