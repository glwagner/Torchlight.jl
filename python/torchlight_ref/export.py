"""HDF5 fixture export (Torchlight fixture schema v1).

Layout (all tensors stored in native PyTorch/NumPy row-major order, so Julia's
HDF5.jl sees axes reversed; dtype is preserved, never forced to float32):

    /meta                       group with attributes (see write_meta)
    /input                      (batch, in)
    /target                     (batch, out)
    /output                     (batch, out)
    /state_dict/<key>           one dataset per state_dict entry
    /intermediates/<name>       named intermediate tensors
    /probes/output_cotangent    (batch, out)
    /probes/input_direction     (batch, in)
    /probes/param_directions/<key>
    /probes/dropout_masks/layer{i}          (train-mode fixtures only)
    /derivatives/loss, grad_input, grad_params/<key>
    /derivatives/vjp_input, vjp_params/<key>
    /derivatives/jvp_output_from_input, jvp_output_from_params
    /training/sgd/...  /training/adam/...   (see write_training)

The layout of /input, /state_dict, /output is compatible with Luximm.jl's
parity fixtures so that ``Luximm.Interop.read_parity`` can read the float32
subset; the extra groups are Torchlight additions.
"""

from __future__ import annotations

import copy
import hashlib
import os
import platform
import sys
from typing import Mapping

import h5py
import numpy as np
import torch

from . import SCHEMA_VERSION


def to_numpy(x: torch.Tensor) -> np.ndarray:
    # NOTE: no .float() here, unlike Luximm's _dump_common.to_numpy.
    return x.detach().cpu().contiguous().numpy()


def _write_dict(group: h5py.Group, name: str, d: Mapping[str, torch.Tensor]):
    g = group.create_group(name)
    for k, v in d.items():
        g.create_dataset(k, data=to_numpy(v))
    return g


def write_meta(f: h5py.File, *, model, model_id: str, seed: int, dtype: torch.dtype,
               mode: str, source_framework: str = "torch", extra: Mapping | None = None):
    g = f.create_group("meta")
    a = g.attrs
    a["schema_version"] = SCHEMA_VERSION
    a["source_framework"] = source_framework
    a["source_version"] = str(torch.__version__)
    a["python_version"] = sys.version.split()[0]
    a["platform"] = platform.platform()
    a["model_id"] = model_id
    a["seed"] = int(seed)
    a["dtype"] = {torch.float32: "float32", torch.float64: "float64"}[dtype]
    a["mode"] = mode
    a["widths"] = np.asarray(model.widths, dtype=np.int64)
    a["n_layers"] = int(model.n_layers)
    a["activation"] = "tanh"
    a["output_activation"] = "identity"
    a["dropout_p"] = float(model.dropout_p)
    a.create("layer_keys", data=np.array(model.layer_keys, dtype=object), dtype=h5py.string_dtype())
    a["input_layout"] = "(batch, features)"
    a["dense_weight_layout"] = "(out, in)"   # nn.Linear.weight logical axes
    a["dense_bias_layout"] = "(out,)"
    a["loss"] = "mse_mean"
    if extra:
        for k, v in extra.items():
            a[k] = v
    return g


def named_params(model) -> dict[str, torch.Tensor]:
    return {k: v for k, v in model.named_parameters()}


def compute_derivatives(model, x, target, cotangent, input_dir, param_dirs, masks=None):
    """Return a dict of derivative tensors for the Torchlight schema.

    All derivatives are computed with the model in its *current* mode; callers
    pass ``masks`` for reproducible train-mode results.
    """
    from .column_mlp import mse_mean

    params = named_params(model)
    names = list(params.keys())
    if set(param_dirs.keys()) != set(names):
        raise ValueError("param_dirs keys must exactly match model.named_parameters(): "
                         f"missing {set(names) - set(param_dirs)}, extra {set(param_dirs) - set(names)}")

    def fwd_functional(xx, *plist):
        pdict = dict(zip(names, plist))
        return torch.func.functional_call(model, pdict, (xx, masks))[0]

    x_ = x.detach().clone().requires_grad_(True)
    plist = [p.detach().clone().requires_grad_(True) for p in params.values()]

    # MSE gradient wrt input and params
    y = fwd_functional(x_, *plist)
    loss = mse_mean(y, target)
    grads = torch.autograd.grad(loss, [x_, *plist])
    out = {
        "loss": loss.detach(),
        "grad_input": grads[0].detach(),
        "grad_params": {n: g.detach() for n, g in zip(names, grads[1:])},
    }

    # VJP with an arbitrary cotangent: d/d(x,theta) sum(v * f)
    y2 = fwd_functional(x_, *plist)
    s = torch.sum(cotangent * y2)
    vgrads = torch.autograd.grad(s, [x_, *plist])
    out["vjp_input"] = vgrads[0].detach()
    out["vjp_params"] = {n: g.detach() for n, g in zip(names, vgrads[1:])}

    # JVPs: forward-mode, separately from an input direction and a param direction
    zeros_p = [torch.zeros_like(p) for p in plist]
    pd = [param_dirs[n] for n in names]
    _, jvp_x = torch.func.jvp(fwd_functional, (x_.detach(), *[p.detach() for p in plist]),
                              (input_dir, *zeros_p))
    _, jvp_p = torch.func.jvp(fwd_functional, (x_.detach(), *[p.detach() for p in plist]),
                              (torch.zeros_like(x_), *pd))
    out["jvp_output_from_input"] = jvp_x.detach()
    out["jvp_output_from_params"] = jvp_p.detach()
    return out


def write_derivatives(f: h5py.File, d: Mapping):
    g = f.create_group("derivatives")
    g.create_dataset("loss", data=to_numpy(d["loss"]))
    g.create_dataset("grad_input", data=to_numpy(d["grad_input"]))
    _write_dict(g, "grad_params", d["grad_params"])
    g.create_dataset("vjp_input", data=to_numpy(d["vjp_input"]))
    _write_dict(g, "vjp_params", d["vjp_params"])
    g.create_dataset("jvp_output_from_input", data=to_numpy(d["jvp_output_from_input"]))
    g.create_dataset("jvp_output_from_params", data=to_numpy(d["jvp_output_from_params"]))
    return g


def one_step(model, x, target, opt_factory, masks=None):
    """Run exactly one optimizer step on a deep copy of ``model``.

    Returns (loss, grads, params_before, params_after, optimizer_state).
    """
    from .column_mlp import mse_mean

    m = copy.deepcopy(model)
    opt = opt_factory(m.parameters())
    before = {k: v.detach().clone() for k, v in m.named_parameters()}
    opt.zero_grad()
    y, _ = m(x, masks)
    loss = mse_mean(y, target)
    loss.backward()
    grads = {k: v.grad.detach().clone() for k, v in m.named_parameters()}
    opt.step()
    after = {k: v.detach().clone() for k, v in m.named_parameters()}
    state = {}
    for k, p in m.named_parameters():
        s = opt.state.get(p, {})
        state[k] = {sk: (sv.detach().clone() if torch.is_tensor(sv) else sv) for sk, sv in s.items()}
    return loss.detach(), grads, before, after, state


def write_training(f: h5py.File, model, x, target, *, lr_sgd=0.05, lr_adam=1e-3,
                   betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, masks=None):
    tg = f.create_group("training")

    loss, grads, before, after, _ = one_step(
        model, x, target, lambda ps: torch.optim.SGD(ps, lr=lr_sgd), masks)
    g = tg.create_group("sgd")
    g.attrs["lr"] = float(lr_sgd)
    g.attrs["steps"] = 1
    g.create_dataset("loss", data=to_numpy(loss))
    _write_dict(g, "grads", grads)
    _write_dict(g, "params_before", before)
    _write_dict(g, "params_after", after)

    loss, grads, before, after, state = one_step(
        model, x, target,
        lambda ps: torch.optim.Adam(ps, lr=lr_adam, betas=betas, eps=eps,
                                    weight_decay=weight_decay),
        masks)
    g = tg.create_group("adam")
    g.attrs["lr"] = float(lr_adam)
    g.attrs["beta1"] = float(betas[0])
    g.attrs["beta2"] = float(betas[1])
    g.attrs["eps"] = float(eps)
    g.attrs["weight_decay"] = float(weight_decay)
    g.attrs["steps"] = 1
    g.attrs["update_rule"] = ("torch.optim.Adam: m_t = b1*m + (1-b1)*g; v_t = b2*v + (1-b2)*g^2; "
                             "mhat = m_t/(1-b1^t); vhat = v_t/(1-b2^t); p -= lr * mhat / (sqrt(vhat) + eps)")
    g.create_dataset("loss", data=to_numpy(loss))
    _write_dict(g, "grads", grads)
    _write_dict(g, "params_before", before)
    _write_dict(g, "params_after", after)
    og = g.create_group("opt_state")
    for k, s in state.items():
        kg = og.create_group(k)
        for sk, sv in s.items():
            if sk == "step":
                # torch stores the step count as a float32 0-D tensor; it is a
                # counter, so record it as an integer attribute.
                kg.attrs["step"] = int(sv.item() if torch.is_tensor(sv) else sv)
            elif torch.is_tensor(sv):
                kg.create_dataset(sk, data=to_numpy(sv))
            else:
                kg.attrs[sk] = sv
    return tg


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_column_mlp_fixture(path: str, *, model, x, target, mode: str, model_id: str,
                              seed: int, masks=None, cotangent=None, input_dir=None,
                              param_dirs=None, gen: torch.Generator | None = None,
                              training: bool = True, extra_meta: Mapping | None = None):
    """Write a complete schema-v1 fixture for a ColumnMLP."""
    from .column_mlp import mse_mean

    dtype = x.dtype
    if mode == "eval":
        model.eval()
        if masks is not None:
            raise ValueError("eval-mode fixtures must not carry dropout masks")
    elif mode == "train":
        model.train()
        if masks is None:
            raise ValueError("train-mode fixtures require explicit dropout masks")
    else:
        raise ValueError(mode)

    if gen is None:
        gen = torch.Generator().manual_seed(seed + 1000)
    with torch.no_grad():
        y, inter = model(x, masks)
    if cotangent is None:
        cotangent = torch.randn(y.shape, dtype=dtype, generator=gen)
    if input_dir is None:
        input_dir = torch.randn(x.shape, dtype=dtype, generator=gen)
    if param_dirs is None:
        param_dirs = {k: torch.randn(v.shape, dtype=dtype, generator=gen)
                      for k, v in model.named_parameters()}

    derivs = compute_derivatives(model, x, target, cotangent, input_dir, param_dirs, masks)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    with h5py.File(path, "w") as f:
        write_meta(f, model=model, model_id=model_id, seed=seed, dtype=dtype, mode=mode,
                   extra=extra_meta)
        f.create_dataset("input", data=to_numpy(x))
        f.create_dataset("target", data=to_numpy(target))
        f.create_dataset("output", data=to_numpy(y))
        _write_dict(f, "state_dict", model.state_dict())
        _write_dict(f, "intermediates", inter)
        pg = f.create_group("probes")
        pg.create_dataset("output_cotangent", data=to_numpy(cotangent))
        pg.create_dataset("input_direction", data=to_numpy(input_dir))
        _write_dict(pg, "param_directions", param_dirs)
        if masks is not None:
            _write_dict(pg, "dropout_masks", {f"layer{i}": m for i, m in enumerate(masks)})
        write_derivatives(f, derivs)
        if training:
            write_training(f, model, x, target, masks=masks)
        if mode == "eval":
            write_domain(f, model, x, gen, dtype)
    return path


def domain_cases(model, x, gen: torch.Generator, dtype):
    """Input families for declared-domain forward validation (plan item 5).

    Returns an ordered dict name -> input tensor.  Every family is evaluated by
    the source model in eval mode so the Julia side can compare over the whole
    declared domain rather than one random batch.
    """
    batch, nin = x.shape
    cases = {}
    cases["zeros_b1"] = torch.zeros(1, nin, dtype=dtype)
    cases["ones_b1"] = torch.ones(1, nin, dtype=dtype)
    cases["constant_minus2_b4"] = torch.full((4, nin), -2.0, dtype=dtype)
    cases["basis"] = torch.eye(nin, dtype=dtype)              # one-hot features
    cases["ramp_b3"] = torch.linspace(-1, 1, 3 * nin, dtype=dtype).reshape(3, nin)
    for scale in (0.01, 1.0, 10.0, 100.0):
        cases[f"randn_scale{scale:g}_b8"] = scale * torch.randn(8, nin, dtype=dtype, generator=gen)
    for b in (1, 2, 7, 32, 256):
        cases[f"randn_b{b}"] = torch.randn(b, nin, dtype=dtype, generator=gen)
    # Column independence: a permuted batch must give a permuted output.
    perm = torch.randperm(batch, generator=gen)
    cases["permuted_reference_batch"] = x[perm]
    cases["reference_batch_first_half"] = x[: batch // 2]
    cases["reference_batch_second_half"] = x[batch // 2:]
    # Saturation / extreme magnitudes (finite but far outside O(1)).
    cases["large_1e3_b2"] = 1e3 * torch.randn(2, nin, dtype=dtype, generator=gen)
    return cases


def write_domain(f: h5py.File, model, x, gen, dtype):
    g = f.create_group("domain")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for name, xi in domain_cases(model, x, gen, dtype).items():
            yi, _ = model(xi, None)
            cg = g.create_group(name)
            cg.create_dataset("input", data=to_numpy(xi))
            cg.create_dataset("output", data=to_numpy(yi))
    if was_training:
        model.train()
    return g
