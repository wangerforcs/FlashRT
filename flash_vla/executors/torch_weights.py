"""Torch-side WeightLoader concretes (stage 7.2).

Concrete ``WeightSource`` / ``WeightSink`` / ``Transform`` / composite
``Source`` implementations that let a ``ModelWeightSpec`` actually load
weights from safetensors into a torch frontend. The runner
(``WeightLoader.run``) is implemented here as well — the stage-7.1
``weight_loader.py`` still owns the protocols and spec dataclasses.

Design constraints (from docs/v2/stage7_weight_loader.md §2.3):
  * ``quant_fp8`` is the single source of truth for FP8 E4M3 quant; we
    import it from ``core.thor_frontend_utils`` to preserve the
    ``.contiguous()`` invariant introduced in stage 5.1.
  * Fused composites (``FusedQKV`` / ``FusedGateUp``) read extra keys
    via the source — they exist as *named* composites rather than ad-hoc
    transform chains so spec files stay readable and mis-wiring is
    caught by the composite itself.
  * Transforms are pure; no transform mutates the source or target. The
    only side channel is ``ctx.scratch['_pending_scale']`` which
    ``Quant`` writes and the runner consumes.
  * Bit-identical guarantee (stage 7.3 rollout) requires transforms to
    reproduce exactly the current hand-written op order.
"""

from __future__ import annotations

from typing import Any

import torch

from flash_vla.core.thor_frontend_utils import (
    interleave_qk as _interleave_qk_core,
    quant_fp8 as _quant_fp8_core,
)

from flash_vla.executors.weight_loader import (
    Item,
    LayerBlock,
    LoaderContext,
    ModelWeightSpec,
    WeightLoader,
)


_FP16 = torch.float16
_FP32 = torch.float32


# ════════════════════════════════════════════════════════════════════
#  Sources
# ════════════════════════════════════════════════════════════════════

class SafetensorsSource:
    """WeightSource backed by a ``safetensors.safe_open`` handle.

    Tensors are fetched lazily on each ``get``. Callers get a torch
    tensor on CUDA (the handle is opened with ``device='cuda'``). No
    caching — each ``get`` returns a fresh view; use transforms to
    shape/quantize.
    """

    def __init__(self, path: str, *, device: str = "cuda"):
        from safetensors import safe_open  # deferred: host python may lack torch+safetensors
        self._sf = safe_open(str(path), framework="pt", device=device)
        self._path = str(path)
        # safetensors' safe_open exposes ``keys()`` — cache once for ``has``.
        self._keys = set(self._sf.keys())

    def get(self, key: str):
        return self._sf.get_tensor(key)

    def has(self, key: str) -> bool:
        return key in self._keys


class DictSource:
    """WeightSource backed by an in-memory ``dict[str, Tensor]``.

    Used by JAX/Orbax path (stage 7.5) and by unit tests. Kept here
    rather than in ``weight_loader.py`` because its only consumers are
    framework-level adapters.
    """

    def __init__(self, d: dict):
        self._d = d

    def get(self, key: str):
        return self._d[key]

    def has(self, key: str) -> bool:
        return key in self._d


# ════════════════════════════════════════════════════════════════════
#  Composite sources (named; resolve to a single tensor)
# ════════════════════════════════════════════════════════════════════

class Cat:
    """Concatenate ``torch.cat`` across N keys along ``dim``.

    Each key is fetched via ``ctx.source.get`` after prefix substitution;
    tensors are cast to fp16 before cat (matches current frontends'
    behaviour where concatenation always follows ``.to(fp16)`` casts).
    Set ``dtype=None`` to skip the cast.
    """

    def __init__(self, keys: list[str], *, dim: int = 0, dtype=_FP16):
        self.keys = keys
        self.dim = dim
        self.dtype = dtype

    def resolve(self, ctx: LoaderContext):
        parts = [ctx.source.get(ctx.subkey(k)) for k in self.keys]
        if self.dtype is not None:
            parts = [p.to(self.dtype) for p in parts]
        return torch.cat(parts, dim=self.dim)


class FusedQKV:
    """Compose a fused QKV tensor, optionally fusing a norm weight and/or
    applying RoPE pair-interleave on Q and K.

    Matches the exact op order of Pi0.5 encoder's QKV loader:

        q_f32 = interleave_qk(source[q], q_heads).float()  # if interleave set
        k_f32 = interleave_qk(source[k], k_heads).float()
        v_f32 = source[v].float()
        if norm_fuse:
            fa = 1.0 + source[norm_fuse].float()
            q_f32 = q_f32 * fa.unsqueeze(0)
            k_f32 = k_f32 * fa.unsqueeze(0)
            v_f32 = v_f32 * fa.unsqueeze(0)
        return torch.cat([q, k, v], dim=0).to(fp16)

    ``interleave_q_heads`` / ``interleave_k_heads`` are optional — omit
    both to skip RoPE interleave (SigLIP/AE paths).
    """

    def __init__(
        self,
        *,
        q: str,
        k: str,
        v: str,
        norm_fuse: str | None = None,
        interleave_q_heads: int | None = None,
        interleave_k_heads: int | None = None,
    ):
        self.q = q
        self.k = k
        self.v = v
        self.norm_fuse = norm_fuse
        self.iq = interleave_q_heads
        self.ik = interleave_k_heads

    def resolve(self, ctx: LoaderContext):
        q = ctx.source.get(ctx.subkey(self.q))
        k = ctx.source.get(ctx.subkey(self.k))
        v = ctx.source.get(ctx.subkey(self.v))
        q_f = _interleave_qk_core(q.float(), self.iq) if self.iq else q.float()
        k_f = _interleave_qk_core(k.float(), self.ik) if self.ik else k.float()
        v_f = v.float()
        if self.norm_fuse is not None:
            fa = 1.0 + ctx.source.get(ctx.subkey(self.norm_fuse)).float()
            q_f = q_f * fa.unsqueeze(0)
            k_f = k_f * fa.unsqueeze(0)
            v_f = v_f * fa.unsqueeze(0)
        return torch.cat([q_f, k_f, v_f], dim=0).to(_FP16)


class FusedGateUp:
    """Compose a fused [gate; up] tensor with optional post-attn norm fuse.

    Matches Pi0.5 encoder's gate_up loader op order:

        ff = 1.0 + source[norm_fuse].float()           # if norm_fuse
        gw = (source[gate].float() * ff.unsqueeze(0)).to(fp16)
        uw = (source[up].float()   * ff.unsqueeze(0)).to(fp16)
        return torch.cat([gw, uw], dim=0)             # [2H, D]

    With ``norm_fuse=None`` the gate/up tensors are read and cast to fp16
    directly before cat — matches decoder/AE path where there is no
    norm fuse and the original code is ``torch.cat([gw, uw], dim=0)``.
    """

    def __init__(self, *, gate: str, up: str, norm_fuse: str | None = None):
        self.gate = gate
        self.up = up
        self.norm_fuse = norm_fuse

    def resolve(self, ctx: LoaderContext):
        if self.norm_fuse is not None:
            ff = 1.0 + ctx.source.get(ctx.subkey(self.norm_fuse)).float()
            ff_u = ff.unsqueeze(0)
            gw = (ctx.source.get(ctx.subkey(self.gate)).float() * ff_u).to(_FP16)
            uw = (ctx.source.get(ctx.subkey(self.up)).float() * ff_u).to(_FP16)
        else:
            gw = ctx.source.get(ctx.subkey(self.gate)).to(_FP16)
            uw = ctx.source.get(ctx.subkey(self.up)).to(_FP16)
        return torch.cat([gw, uw], dim=0)


# ════════════════════════════════════════════════════════════════════
#  Transforms
# ════════════════════════════════════════════════════════════════════

class ToFp16:
    def apply(self, x, ctx):
        return x.to(_FP16)


class ToFp32:
    def apply(self, x, ctx):
        return x.to(_FP32)


class T:
    """``.T.contiguous()`` — CUTLASS FP8 col-major path (encoder GEMMs)."""

    def apply(self, x, ctx):
        return x.T.contiguous()


class tT:
    """``.t().contiguous()`` — cuBLASLt FP8 path (decoder/AE GEMMs).

    Numerically identical to ``T`` for 2-D tensors but spelled
    differently in the existing code; keeping the two classes distinct
    preserves code-review diffability against the original loader."""

    def apply(self, x, ctx):
        return x.t().contiguous()


class InterleaveQK:
    """Standalone RoPE pair-interleave (use when not inside ``FusedQKV``)."""

    def __init__(self, num_heads: int):
        self.num_heads = num_heads

    def apply(self, x, ctx):
        return _interleave_qk_core(x, self.num_heads)


class Quant:
    """FP8 E4M3 per-tensor quant. Stashes scale into ``ctx.scratch``.

    The runner consumes ``ctx.scratch['_pending_scale']`` after the
    transform pipeline finishes, and (if ``item.scale_into`` is set)
    appends it to ``ctx.scales[scale_into]`` in spec order. This keeps
    transforms tensor-in/tensor-out while still propagating the quant
    scale.
    """

    def apply(self, x, ctx):
        fp8, s = _quant_fp8_core(x)
        ctx.scratch["_pending_scale"] = float(s)
        return fp8


class Mul:
    """Multiply tensor by a scalar. Used for action_out_proj weight
    (which bakes ``-1/steps`` into the weight/bias)."""

    def __init__(self, factor: float):
        self.factor = factor

    def apply(self, x, ctx):
        return x * self.factor


# ════════════════════════════════════════════════════════════════════
#  Sinks
# ════════════════════════════════════════════════════════════════════

class Attr:
    """Assign the final tensor to ``target.<name>`` once.

    Used for singletons (embedding, postln, projector, time_mlp, etc.).
    """

    def __init__(self, name: str):
        self.name = name
        self._bound_target = None
        self._stored = False

    def _bind(self, target):
        self._bound_target = target

    def store(self, tensor, *, scale=None):
        if self._stored:
            raise RuntimeError(f"Attr sink '{self.name}' stored twice")
        setattr(self._bound_target, self.name, tensor)
        self._stored = True

    def finalize(self) -> None:
        return None


class TensorList:
    """Append each iteration's tensor into ``target.<name>`` (a list).

    List is created on first ``store`` if not already an attribute. The
    sink holds no state between runs — each ``WeightLoader.run`` call
    starts fresh.
    """

    def __init__(self, name: str):
        self.name = name
        self._bound_target = None
        self._list: list | None = None

    def _bind(self, target):
        self._bound_target = target
        self._list = []
        setattr(target, self.name, self._list)

    def store(self, tensor, *, scale=None):
        assert self._list is not None, "TensorList not bound"
        self._list.append(tensor)

    def finalize(self) -> None:
        return None


class FlatCat:
    """Append each iteration then ``torch.cat([w.reshape(-1) for w in ...])``
    at finalize. Stores the flat tensor as ``target.<name>``.

    Matches Pi0.5 decoder's per-layer flat weight tensors (``_dec_qkv_flat``).
    """

    def __init__(self, name: str):
        self.name = name
        self._bound_target = None
        self._parts: list | None = None

    def _bind(self, target):
        self._bound_target = target
        self._parts = []

    def store(self, tensor, *, scale=None):
        assert self._parts is not None, "FlatCat not bound"
        self._parts.append(tensor)

    def finalize(self) -> None:
        flat = torch.cat([w.reshape(-1) for w in self._parts])
        setattr(self._bound_target, self.name, flat)


# Runner moved to ``weight_loader.py`` (stage 7.5) — framework-agnostic.

__all__ = [
    "SafetensorsSource",
    "DictSource",
    "Cat",
    "FusedQKV",
    "FusedGateUp",
    "ToFp16",
    "ToFp32",
    "T",
    "tT",
    "InterleaveQK",
    "Quant",
    "Mul",
    "Attr",
    "TensorList",
    "FlatCat",
    "WeightLoader",
]
