"""Re-evaluate a Torchlight dense/tanh fixture using JAX on CPU.

This is an independent reference evaluator for the declared column MLP, not an
architecture converter. All tensors retain the fixture's PyTorch logical axes,
including weights shaped (output, input). No PyTorch executable is imported.

Usage: python -m torchlight_ref.jax_reference --input torch.h5 --out jax.h5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import jax

# Set precision before constructing any arrays. CPU is selected explicitly below.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _array(node, dtype, shape=None):
    value = np.asarray(node[()])
    if value.dtype != dtype:
        raise ValueError(f"{node.name}: dtype {value.dtype} != declared {dtype}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{node.name}: shape {value.shape} != expected {shape}")
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError(f"{node.name}: expected nonempty finite array")
    return value


def _forward(params, x, layer_keys):
    intermediates = {}
    for index, key in enumerate(layer_keys):
        affine = x @ params[f"{key}.weight"].T + params[f"{key}.bias"]
        intermediates[f"layer{index}_preact"] = affine
        x = jnp.tanh(affine) if index < len(layer_keys) - 1 else affine
        if index < len(layer_keys) - 1:
            intermediates[f"layer{index}_act"] = x
    return x, intermediates


def _write(group, key, value):
    value = np.asarray(value)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"JAX produced nonfinite values for {key}")
    group.create_dataset(key, data=value)


def export_jax_reference(input_path, output_path):
    """Write independently evaluated forward/AD evidence from a schema-v1 fixture.

    Required model metadata: ``activation=tanh`` and ordered ``layer_keys`` naming
    the ordered linear modules. Only eval-mode, biased dense/tanh MLPs are accepted.
    Source training snapshots are intentionally excluded: this adapter does not
    claim optimizer or stochastic-state parity.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output must be different files")
    source_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    with h5py.File(input_path, "r") as source:
        attrs = dict(source["meta"].attrs)
        if int(attrs["schema_version"]) != 1:
            raise ValueError("only fixture schema_version=1 is supported")
        if _text(attrs["mode"]) != "eval":
            raise ValueError("only eval-mode MLP fixtures are supported")
        if "dropout_masks" in source["probes"]:
            raise ValueError("eval fixtures with explicit dropout masks are unsupported")
        if _text(attrs["activation"]) != "tanh":
            raise ValueError("only tanh hidden activations are supported")
        if _text(attrs["output_activation"]) != "identity":
            raise ValueError("only an identity output activation is supported")
        if _text(attrs["loss"]) != "mse_mean":
            raise ValueError("only loss=mse_mean is supported")
        if _text(attrs["dense_weight_layout"]) != "(out, in)":
            raise ValueError("this adapter requires dense_weight_layout=(out, in)")
        raw_keys = attrs["layer_keys"]
        layer_keys = (json.loads(_text(raw_keys)) if isinstance(raw_keys, (str, bytes))
                      else [_text(k) for k in raw_keys])
        if (not isinstance(layer_keys, list) or not layer_keys
                or not all(isinstance(k, str) and k for k in layer_keys)
                or len(layer_keys) != len(set(layer_keys))):
            raise ValueError("layer_keys must be a nonempty ordered list of unique names")
        widths = np.asarray(attrs["widths"])
        if (widths.ndim != 1 or widths.dtype.kind not in "iu" or np.any(widths <= 0)
                or len(widths) != len(layer_keys) + 1
                or int(attrs["n_layers"]) != len(layer_keys)):
            raise ValueError("widths and n_layers must describe the declared layers")
        dtype = np.dtype(_text(attrs["dtype"]))
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("only float32 and float64 fixtures are supported")
        x = _array(source["input"], dtype)
        if x.ndim != 2:
            raise ValueError("input must have (batch, features) axes")
        if x.shape[1] != widths[0]:
            raise ValueError("input feature count does not match widths")
        state = source["state_dict"]
        required_keys = {f"{key}.{leaf}" for key in layer_keys for leaf in ("weight", "bias")}
        if set(state) != required_keys:
            raise ValueError("state_dict must contain exactly the declared weights and biases")
        params = {}
        features = x.shape[1]
        for index, key in enumerate(layer_keys):
            weight = _array(state[f"{key}.weight"], dtype)
            if weight.ndim != 2 or weight.shape[1] != features:
                raise ValueError(f"{key}.weight: expected (output, {features})")
            features = weight.shape[0]
            if features != widths[index + 1]:
                raise ValueError(f"{key}.weight: output count does not match widths")
            params[f"{key}.weight"] = weight
            params[f"{key}.bias"] = _array(state[f"{key}.bias"], dtype, (features,))
        output_shape = (x.shape[0], features)
        target = _array(source["target"], dtype, output_shape)
        cotangent = _array(source["probes/output_cotangent"], dtype, output_shape)
        direction = _array(source["probes/input_direction"], dtype, x.shape)
        param_directions = None
        if "param_directions" in source["probes"]:
            group = source["probes/param_directions"]
            if set(group) != required_keys:
                raise ValueError("param_directions must cover every parameter exactly")
            param_directions = {k: _array(group[k], dtype, v.shape) for k, v in params.items()}

        with jax.default_device(jax.devices("cpu")[0]), jax.default_matmul_precision("highest"):
            ps = jax.tree.map(jnp.asarray, params)
            jx, jt = jnp.asarray(x), jnp.asarray(target)
            predict = lambda p, z: _forward(p, z, layer_keys)[0]
            objective = lambda p, z: jnp.mean(jnp.square(predict(p, z) - jt))
            output, intermediates = _forward(ps, jx, layer_keys)
            loss, (grad_params, grad_input) = jax.value_and_grad(objective, argnums=(0, 1))(ps, jx)
            _, pullback = jax.vjp(predict, ps, jx)
            vjp_params, vjp_input = pullback(jnp.asarray(cotangent))
            _, jvp_output = jax.jvp(lambda z: predict(ps, z), (jx,), (jnp.asarray(direction),))
            param_jvp = None
            if param_directions is not None:
                _, param_jvp = jax.jvp(lambda p: predict(p, jx), (ps,),
                                       (jax.tree.map(jnp.asarray, param_directions),))
            domain_outputs = {}
            if "domain" in source:
                for name, case in source["domain"].items():
                    case_input = _array(case["input"], dtype)
                    if case_input.ndim != 2 or case_input.shape[1] != widths[0]:
                        raise ValueError(f"domain/{name}/input: expected (batch, {widths[0]})")
                    domain_outputs[name] = predict(ps, jnp.asarray(case_input))

        # Write atomically so validation or evaluation failures do not leave a
        # partial reference under the requested output filename.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".jax-reference-", suffix=".h5", dir=output_path.parent)
        os.close(fd)
        try:
            with h5py.File(temporary, "w") as dest:
                meta = dest.create_group("meta")
                for key, value in attrs.items():
                    meta.attrs[key] = value
                meta.attrs["upstream_source_framework"] = _text(attrs["source_framework"])
                meta.attrs["upstream_source_commit"] = _text(attrs.get("source_commit", "unknown"))
                meta.attrs["source_framework"] = "jax"
                meta.attrs["source_version"] = jax.__version__
                meta.attrs["source_commit"] = "adapter-sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                meta.attrs["source_fixture_sha256"] = source_hash
                meta.attrs["jax_version"] = jax.__version__
                meta.attrs["numpy_version"] = np.__version__
                meta.attrs["backend"] = "cpu"
                meta.attrs["matmul_precision"] = "highest"
                meta.attrs["jax_enable_x64"] = True
                meta.attrs["dense_weight_layout"] = "(out, in)"
                meta.attrs["loss_definition"] = "mean((output - target)^2), all elements"
                meta.attrs["training_status"] = "not_tested"
                for key in ("input", "target", "state_dict", "probes"):
                    source.copy(key, dest)
                _write(dest, "output", output)
                stages = dest.create_group("intermediates")
                for key, value in intermediates.items():
                    _write(stages, key, value)
                derivatives = dest.create_group("derivatives")
                for key, value in (("loss", loss), ("grad_input", grad_input),
                                   ("vjp_input", vjp_input), ("jvp_output_from_input", jvp_output)):
                    _write(derivatives, key, value)
                for name, values in (("grad_params", grad_params), ("vjp_params", vjp_params)):
                    group = derivatives.create_group(name)
                    for key, value in values.items():
                        _write(group, key, value)
                if param_jvp is not None:
                    _write(derivatives, "jvp_output_from_params", param_jvp)
                if "domain" in source:
                    domain = dest.create_group("domain")
                    for key, value in source["domain"].attrs.items():
                        domain.attrs[key] = value
                    for name, value in domain_outputs.items():
                        case = domain.create_group(name)
                        for key, attr in source[f"domain/{name}"].attrs.items():
                            case.attrs[key] = attr
                        source.copy(f"domain/{name}/input", case, name="input")
                        _write(case, "output", value)
            os.replace(temporary, output_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="PyTorch schema-v1 fixture")
    parser.add_argument("--out", required=True, type=Path, help="independently evaluated JAX fixture")
    args = parser.parse_args()
    path = export_jax_reference(args.input, args.out)
    print(f"Wrote JAX CPU reference: {path}")


if __name__ == "__main__":
    main()
