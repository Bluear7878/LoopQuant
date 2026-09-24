"""Compact Ouro adapter exposing recurrent invocations to LoopQuant."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .quantization import QUANTIZATION_CONTEXT, QuantizedLinear


def detect_model_family(model_or_path: Any) -> str:
    text = str(model_or_path).lower()
    model_type = str(getattr(getattr(model_or_path, "config", None), "model_type", "")).lower()
    if "ouro" in text or "ouro" in model_type:
        return "ouro"
    raise ValueError("unable to identify a validated looped model family; pass --model-family ouro")


def _patch_ouro_cache_compatibility(model: Any) -> bool:
    """Preserve Ouro's mutable recurrent-cache lists on Transformers 4.55."""
    module = sys.modules.get(type(model).__module__)
    cache_cls = None if module is None else getattr(module, "UniversalTransformerCache", None)
    if cache_cls is None:
        return False
    changed = False
    for name in ("key_cache", "value_cache"):
        inherited = getattr(cache_cls, name, None)
        if isinstance(inherited, property) and inherited.fset is None:
            storage = f"_loopquant_{name}"
            setattr(
                cache_cls,
                name,
                property(
                    lambda instance, key=storage: instance.__dict__.setdefault(key, []),
                    lambda instance, value, key=storage: instance.__dict__.__setitem__(key, value),
                ),
            )
            changed = True
    if "get_mask_sizes" not in cache_cls.__dict__:
        cache_cls.get_mask_sizes = lambda instance, position, layer: (
            int(position.shape[0]) + int(instance.get_seq_length(layer)),
            0,
        )
        changed = True
    if "get_max_cache_shape" not in cache_cls.__dict__:
        cache_cls.get_max_cache_shape = lambda instance, layer_idx=0: -1
        changed = True
    return changed


def register_loop_hooks(model: Any, model_family: str = "ouro") -> None:
    """Track the current recurrent invocation using Ouro's ``current_ut`` argument."""
    if model_family.lower() != "ouro":
        raise ValueError("only the validated Ouro adapter is included")
    core = getattr(model, "model", model)
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("Ouro adapter expected model.model.layers")
    for layer in layers:
        if getattr(layer, "_loopquant_hooked", False):
            continue
        original = layer.forward

        def make_wrapper(function):
            def wrapped(*args: Any, **kwargs: Any):
                step = kwargs.get("current_ut", kwargs.get("loop_idx", kwargs.get("mhc_loop_idx")))
                if step is not None:
                    QUANTIZATION_CONTEXT.current_loop = int(step)
                return function(*args, **kwargs)

            return wrapped

        layer.forward = make_wrapper(original)
        layer._loopquant_hooked = True
    if not getattr(core, "_loopquant_outer_hooked", False):
        original_core_forward = core.forward

        def core_forward(*args: Any, **kwargs: Any):
            QUANTIZATION_CONTEXT.current_loop = 0
            return original_core_forward(*args, **kwargs)

        core.forward = core_forward
        core._loopquant_outer_hooked = True


def get_recurrent_loop_count(model: Any) -> int:
    return int(getattr(model, "_loopquant_num_loops", getattr(model.config, "total_ut_steps", 4)))


def iter_quantized_modules(model: Any):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            yield name, module


def load_looped_transformer(
    model_path: str | Path,
    model_family: str = "ouro",
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    num_loops: int = 4,
):
    """Load Ouro-1.4B or Ouro-2.6B and force exactly ``num_loops`` invocations."""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model checkpoint does not exist: {path}")
    if model_family.lower() != "ouro":
        raise ValueError("this public release validates Ouro-1.4B and Ouro-2.6B only")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.total_ut_steps = int(num_loops)
    config.early_exit_threshold = 1.0
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    _patch_ouro_cache_compatibility(model)
    model.to(device).eval()
    model.config.total_ut_steps = int(num_loops)
    model.config.early_exit_threshold = 1.0
    model.config.use_cache = False
    if hasattr(model, "model"):
        model.model.total_ut_steps = int(num_loops)
    model.early_exit_step = int(num_loops) - 1
    model.early_exit_threshold = 1.0
    original_forward = model.forward

    def forced_forward(*args: Any, **kwargs: Any):
        kwargs.setdefault("exit_at_step", int(num_loops) - 1)
        kwargs.setdefault("exit_threshold", 1.0)
        kwargs.setdefault("use_weighted_exit", False)
        return original_forward(*args, **kwargs)

    model.forward = forced_forward
    model._loopquant_num_loops = int(num_loops)
    model._loopquant_model_path = str(path)
    QUANTIZATION_CONTEXT.configure(int(num_loops))
    register_loop_hooks(model, model_family)
    return model, tokenizer
