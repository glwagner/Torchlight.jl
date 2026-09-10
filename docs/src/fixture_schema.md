# Torchlight fixture schema v1

Fixtures are HDF5 files written by the source framework
(`python/torchlight_ref/export.py`, or the JAX adapter in
`python/torchlight_ref/jax_reference.py`).  Arrays are stored in the source's
row-major order and **dtype is preserved**; HDF5.jl reads them with axes
reversed relative to the source logical order, so `(batch, features)` arrives
as `(features, batch)` (Lux-natural) and an `(out, in)` dense weight arrives as
`(in, out)` and needs `Torchlight.to_lux_dense_weight`.  The `/input`,
`/state_dict`, `/output` subset follows the same layout convention as
Luximm.jl's parity fixtures, so fixtures of that form can be read too.

| Node | Contents |
|---|---|
| `/meta` attrs | `schema_version=1`, `source_framework`, `source_version`, `python_version`, `platform`, `model_id`, `seed`, `dtype` (`float32`/`float64`), `mode` (`eval`/`train`), `widths`, `n_layers`, `activation`, `output_activation`, `dropout_p`, `layer_keys`, `input_layout`, `dense_weight_layout`, `dense_bias_layout`, `loss` |
| `/input`, `/target`, `/output` | `(batch, in)`, `(batch, out)`, `(batch, out)` |
| `/state_dict/<key>` | every source parameter/buffer; `layers.{i}.weight` `(out, in)`, `layers.{i}.bias` `(out,)` |
| `/intermediates/layer{i}_preact`, `layer{i}_act`, (`layer{i}_drop`) | affine output, post-tanh (pre-dropout), post-dropout (train mode) |
| `/probes/output_cotangent`, `input_direction`, `param_directions/<key>` | shared probe directions `v`, `u_x`, `u_θ` |
| `/probes/dropout_masks/layer{i}` | 0/1 masks, train-mode fixtures only |
| `/derivatives/loss`, `grad_input`, `grad_params/<key>` | `mse_mean` and its gradients |
| `/derivatives/vjp_input`, `vjp_params/<key>` | gradients of `sum(v .* f)` |
| `/derivatives/jvp_output_from_input`, `jvp_output_from_params` | `J_x u_x`, `J_θ u_θ` |
| `/training/sgd` | attrs `lr`, `steps`; `loss`, `grads/<key>`, `params_before/<key>`, `params_after/<key>` |
| `/training/adam` | attrs `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `steps`, `update_rule`; datasets as SGD plus `opt_state/<key>/exp_avg`, `exp_avg_sq`, attr `step` |
| `/domain/<case>/input`, `output` | declared-domain input families (see `export.py::domain_cases`) |

`mse_mean` is the mean over all `batch * out` elements of the squared error.
The Julia reader (`Torchlight.read_fixture`) rejects unknown schema versions,
missing required nodes, non-finite floating arrays, and floating arrays whose
element type differs from the declared `dtype`; non-floating buffers are
preserved as stored.
