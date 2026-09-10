"""Independent PyTorch controls and failure injection for the JAX adapter."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from torchlight_ref.jax_reference import export_jax_reference


def make_fixture(path, dtype):
    rng = np.random.default_rng(48)
    widths = [7, 11, 5, 3]
    arrays = {}
    for index, (n_in, n_out) in enumerate(zip(widths, widths[1:])):
        arrays[f"layers.{index}.weight"] = (rng.normal(size=(n_out, n_in)) * 0.2).astype(dtype)
        arrays[f"layers.{index}.bias"] = (rng.normal(size=(n_out,)) * 0.1).astype(dtype)
    x = rng.normal(size=(13, widths[0])).astype(dtype)
    target = rng.normal(size=(13, widths[-1])).astype(dtype)
    cotangent = rng.normal(size=target.shape).astype(dtype)
    direction = rng.normal(size=x.shape).astype(dtype)
    param_directions = {k: rng.normal(size=v.shape).astype(dtype) for k, v in arrays.items()}
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs.update(schema_version=1, source_framework="pytorch",
                          source_version=str(torch.__version__), source_commit="synthetic-test",
                          model_id="column_mlp_tiny", seed=48, dtype=np.dtype(dtype).name,
                          mode="eval", widths=widths, n_layers=3, activation="tanh",
                          output_activation="identity", dropout_p=0.1, loss="mse_mean",
                          layer_keys=[f"layers.{i}" for i in range(3)])
        f["input"], f["target"] = x, target
        for key, value in arrays.items():
            f[f"state_dict/{key}"] = value
            f[f"probes/param_directions/{key}"] = param_directions[key]
        f["probes/output_cotangent"] = cotangent
        f["probes/input_direction"] = direction
        # Poison source results: the JAX fixture must evaluate, never copy them.
        f["output"] = np.full_like(target, 12345)
        f["derivatives/grad_input"] = np.full_like(x, 12345)
        f["training/sgd/loss"] = 12345.0
    return arrays, x, target, cotangent, direction, param_directions


def torch_reference(arrays, x, target, cotangent, direction, param_directions):
    keys = list(arrays)
    params = tuple(torch.tensor(arrays[k], requires_grad=True) for k in keys)
    tx = torch.tensor(x, requires_grad=True)
    tt = torch.tensor(target)

    def forward(z, *parameters):
        for index in range(0, len(parameters), 2):
            z = torch.nn.functional.linear(z, parameters[index], parameters[index + 1])
            if index < len(parameters) - 2:
                z = torch.tanh(z)
        return z

    output = forward(tx, *params)
    loss = torch.mean((output - tt) ** 2)
    grads = torch.autograd.grad(loss, (tx, *params), retain_graph=True)
    vjps = torch.autograd.grad(torch.sum(output * torch.tensor(cotangent)), (tx, *params))
    _, input_jvp = torch.autograd.functional.jvp(lambda z: forward(z, *params), tx,
                                                torch.tensor(direction))
    _, param_jvp = torch.autograd.functional.jvp(lambda *p: forward(tx, *p), params,
                                                tuple(torch.tensor(param_directions[k]) for k in keys))
    result = {"output": output, "derivatives/loss": loss,
              "derivatives/grad_input": grads[0], "derivatives/vjp_input": vjps[0],
              "derivatives/jvp_output_from_input": input_jvp,
              "derivatives/jvp_output_from_params": param_jvp}
    for index, key in enumerate(keys):
        result[f"derivatives/grad_params/{key}"] = grads[index + 1]
        result[f"derivatives/vjp_params/{key}"] = vjps[index + 1]
    return {k: v.detach().numpy() for k, v in result.items()}


class JaxReferenceTests(unittest.TestCase):
    def test_independent_forward_and_derivatives(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                source, dest = Path(tmp) / "source.h5", Path(tmp) / "jax.h5"
                inputs = make_fixture(source, dtype)
                expected = torch_reference(*inputs)
                export_jax_reference(source, dest)
                with h5py.File(dest, "r") as f:
                    self.assertEqual(f["meta"].attrs["source_framework"], "jax")
                    self.assertEqual(f["meta"].attrs["training_status"], "not_tested")
                    self.assertNotIn("training", f)
                    self.assertEqual(len(f["meta"].attrs["source_fixture_sha256"]), 64)
                    self.assertEqual(set(f["intermediates"]), {
                        "layer0_preact", "layer0_act", "layer1_preact", "layer1_act", "layer2_preact"})
                    for key, reference in expected.items():
                        actual = f[key][()]
                        self.assertEqual(actual.dtype, reference.dtype, key)
                        self.assertEqual(actual.shape, reference.shape, key)
                        atol, rtol = ((1e-6, 1e-4) if dtype == np.float32 else (1e-10, 1e-8))
                        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, err_msg=key)

    def test_rejects_unsupported_semantics(self):
        mutations = [("mode", "train"), ("activation", "relu"),
                     ("output_activation", "tanh"), ("loss", "mse_sum"),
                     ("widths", [7, 12, 5, 3]), ("n_layers", 4),
                     ("layer_keys", ["layers.0", "layers.0", "layers.2"])]
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                source, dest = Path(tmp) / "source.h5", Path(tmp) / "jax.h5"
                make_fixture(source, np.float64)
                with h5py.File(source, "r+") as f:
                    f["meta"].attrs[key] = value
                with self.assertRaises(ValueError):
                    export_jax_reference(source, dest)
                self.assertFalse(dest.exists())

    def test_rejects_invalid_arrays_and_mapping(self):
        for defect in ("dtype", "shape", "nonfinite", "extra_key", "missing_key", "partial_direction", "eval_mask"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                source, dest = Path(tmp) / "source.h5", Path(tmp) / "jax.h5"
                make_fixture(source, np.float64)
                with h5py.File(source, "r+") as f:
                    if defect == "dtype":
                        value = f["input"][()].astype(np.float32)
                        del f["input"]
                        f["input"] = value
                    elif defect == "shape":
                        del f["target"]
                        f["target"] = np.zeros((1, 3), dtype=np.float64)
                    elif defect == "nonfinite":
                        f["input"][0, 0] = np.nan
                    elif defect == "extra_key":
                        f["state_dict/unexpected"] = np.ones(2)
                    elif defect == "missing_key":
                        del f["state_dict/layers.0.bias"]
                    elif defect == "partial_direction":
                        del f["probes/param_directions/layers.0.bias"]
                    else:
                        f["probes/dropout_masks/layer0"] = np.ones((13, 11))
                with self.assertRaises(ValueError):
                    export_jax_reference(source, dest)
                self.assertFalse(dest.exists())

    def test_preserves_source_and_existing_output_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source.h5", Path(tmp) / "jax.h5"
            make_fixture(source, np.float64)
            original = source.read_bytes()
            with self.assertRaises(ValueError):
                export_jax_reference(source, source)
            self.assertEqual(original, source.read_bytes())
            dest.write_bytes(b"existing output")
            with h5py.File(source, "r+") as f:
                f["meta"].attrs["mode"] = "train"
            with self.assertRaises(ValueError):
                export_jax_reference(source, dest)
            self.assertEqual(dest.read_bytes(), b"existing output")


if __name__ == "__main__":
    unittest.main()
