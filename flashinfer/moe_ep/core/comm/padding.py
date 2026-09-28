"""Explicit LL wire padding; never pad the logical compute problem."""

from __future__ import annotations


def pad_ll_rows(x, width: int):
    """Zero-pad the last dimension without changing existing BF16 bits."""
    import torch

    if x.shape[-1] > width:
        raise ValueError(f"transport width {width} is smaller than row {x.shape[-1]}")
    if x.shape[-1] == width:
        return x
    if x.dtype != torch.bfloat16:
        raise ValueError("LL row padding requires BF16 tensors")
    padded = torch.zeros((*x.shape[:-1], width), dtype=x.dtype, device=x.device)
    padded[..., : x.shape[-1]].copy_(x)
    return padded


def prepare_ll_combine(x, out, params, num_tokens: int):
    """Return wire input/output and logical output; caller copies after completion."""
    import torch

    hidden = params.token_hidden_size
    if x.shape[-1] != hidden:
        raise ValueError(f"combine expects logical width {hidden}, got {x.shape[-1]}")
    if out is None:
        out = torch.empty(num_tokens, hidden, dtype=x.dtype, device=x.device)
    elif (
        tuple(out.shape) != (num_tokens, hidden)
        or out.dtype != x.dtype
        or out.device != x.device
    ):
        raise ValueError(
            "combine out must match logical token shape, dtype, and device"
        )
    width = params.combine_hidden_size
    if width == hidden:
        return x, out, out
    wire_x = pad_ll_rows(x, width)
    wire_out = torch.empty(num_tokens, width, dtype=x.dtype, device=x.device)
    return wire_x, wire_out, out


def finish_ll_combine(wire_out, out):
    if wire_out is not out:
        out.copy_(wire_out[..., : out.shape[-1]])
    return out
