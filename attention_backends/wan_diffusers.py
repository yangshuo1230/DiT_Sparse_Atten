"""SVG-style Diffusers Wan execution with study-owned attention routing."""

from __future__ import annotations

import functools
import math
import time
import types

import torch

from . import context as runtime_context
from .svg_ops import (
    apply_rope,
    gated_residual,
    layer_norm,
    modulate,
    operator_status,
    rms_norm,
)


class StudyWanProcessor:
    """SVG processor structure with its attention core delegated to study."""

    def __init__(self, backend, layer_id):
        self.backend = backend
        self.layer_id = layer_id

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        timestep=None,
    ):
        del timestep
        if encoder_hidden_states is not None:
            raise ValueError("StudyWanProcessor is only valid for Wan self-attention")

        runtime_context._STATE.attention_id = self.layer_id
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            query = rms_norm(query.contiguous(), attn.norm_q.weight, attn.norm_q.eps)
        if attn.norm_k is not None:
            key = rms_norm(key.contiguous(), attn.norm_k.weight, attn.norm_k.eps)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2).contiguous()
        if rotary_emb is not None:
            query, key = apply_rope(query, key, rotary_emb)

        output = self.backend(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            dtype=query.dtype,
            attention_mask=attention_mask,
        )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        output = attn.to_out[1](output)
        return output


def _block_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    temb,
    rotary_emb,
    timestep=None,
):
    shifts = (self.scale_shift_table + temb.float()).chunk(6, dim=1)
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = shifts

    norm_hidden_states = layer_norm(
        hidden_states, self.norm1.weight, self.norm1.bias, self.norm1.eps
    )
    norm_hidden_states = modulate(
        norm_hidden_states, scale_msa, shift_msa, hidden_states.dtype
    )
    attention_output = self.attn1(
        hidden_states=norm_hidden_states, rotary_emb=rotary_emb, timestep=timestep
    )
    hidden_states = gated_residual(
        hidden_states, attention_output, gate_msa, hidden_states.dtype
    )

    norm_hidden_states = layer_norm(
        hidden_states, self.norm2.weight, self.norm2.bias, self.norm2.eps
    )
    attention_output = self.attn2(
        hidden_states=norm_hidden_states.type_as(hidden_states),
        encoder_hidden_states=encoder_hidden_states,
    )
    hidden_states = hidden_states + attention_output

    norm_hidden_states = layer_norm(
        hidden_states, self.norm3.weight, self.norm3.bias, self.norm3.eps
    )
    norm_hidden_states = modulate(
        norm_hidden_states, c_scale_msa, c_shift_msa, hidden_states.dtype
    )
    feed_forward_output = self.ffn(norm_hidden_states)
    return gated_residual(
        hidden_states, feed_forward_output, c_gate_msa, hidden_states.dtype
    )


def _model_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    return_dict=True,
    attention_kwargs=None,
):
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

    attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
    lora_scale = attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)

    batch_size, _, num_frames, height, width = hidden_states.shape
    patch_t, patch_h, patch_w = self.config.patch_size
    output_frames = num_frames // patch_t
    output_height = height // patch_h
    output_width = width // patch_w
    runtime_context._STATE.grid_size = (output_frames, output_height, output_width)

    rotary_emb = self.rope(hidden_states)
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image
        )
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states_image, encoder_hidden_states], dim=1
        )

    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            timestep=timestep,
        )

    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (
        self.norm_out(hidden_states.float()) * (1 + scale) + shift
    ).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(
        batch_size,
        output_frames,
        output_height,
        output_width,
        patch_t,
        patch_h,
        patch_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def _timestep_value(timestep):
    if timestep is None:
        return None
    if torch.is_tensor(timestep):
        return float(timestep.detach().flatten()[0].item())
    return float(timestep)


def install_svg_wan_pipeline(transformer, backend):
    """Install SVG execution and study attention into a loaded Wan transformer."""
    if getattr(transformer, "_WAN_ATTENTION_STUDY_SVG", False):
        return operator_status()

    runtime_context.reset_runtime_state()
    for layer_id, block in enumerate(transformer.blocks):
        block.forward = types.MethodType(_block_forward, block)
        block.attn1.set_processor(StudyWanProcessor(backend, layer_id))

    original_model_forward = types.MethodType(_model_forward, transformer)

    @functools.wraps(original_model_forward)
    def timed_forward(instance, *args, **kwargs):
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        timestep_value = _timestep_value(timestep)
        state = runtime_context._STATE
        if timestep_value is not None:
            if state.last_timestep is None or not math.isclose(
                timestep_value, state.last_timestep
            ):
                state.denoising_step += 1
                state.cfg_branch = 0
                state.last_timestep = timestep_value
            else:
                state.cfg_branch += 1

        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        start_event = end_event = None
        if hidden_states is not None and hidden_states.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        wall_started = time.perf_counter()
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            if end_event is not None:
                end_event.record()
            state.forward_calls.append(
                {
                    "step": state.denoising_step,
                    "branch": state.cfg_branch,
                    "timestep": timestep_value,
                    "wall_seconds_unsynchronized": time.perf_counter() - wall_started,
                    "start_event": start_event,
                    "end_event": end_event,
                }
            )

    transformer.forward = types.MethodType(timed_forward, transformer)
    transformer._WAN_ATTENTION_STUDY_SVG = True
    return operator_status()
