# Original JAX comparison and Float32 calibration

These are the original reviewer measurements from 10 September 2026, preserved
without changing the numeric results or acceptance thresholds. They precede the
final fixture metadata and domain extensions. The separate Julia acceptance
snapshots in the parent directory use the later fixture files.

- [jax_comparison.json](jax_comparison.json) compares independently evaluated
  JAX outputs, intermediates, MSE gradients, VJPs, and directional JVPs against
  the PyTorch reference on shared arrays. Each array records its maximum absolute
  error, normalized L2 error, failing component count, and applied tolerances.
- [float32_calibration.json](float32_calibration.json) compares the three failing
  Float32 quantities with a JAX Float64 evaluation of **the same Float32 inputs,
  weights, cotangent, and parameter directions promoted to Float64**. This is
  not a comparison with a separately generated Float64 fixture.
- [provenance.json](provenance.json) records the original source/JAX fixture
  hashes, source and evaluator metadata, evaluator source hashes, and hashes of
  the two result JSON files. The source hashes were checked against the hashes
  embedded in each JAX fixture. They differ from the later acceptance fixtures
  because the fixture format gained additional metadata and domain cases.

Execution used PyTorch 2.14.0, JAX 0.11.1 and NumPy 2.5.3 on CPU. JAX X64 was
enabled before array construction; Float32 cases retained Float32 arrays.

The componentwise acceptance rule was
`abs(actual - reference) <= atol + rtol * abs(reference)`, with `(atol, rtol)`
equal to `(1e-10, 1e-8)` for Float64 and `(1e-6, 1e-4)` for Float32.

| Fixture | Compared arrays | Failing arrays |
| --- | ---: | ---: |
| tiny Float64 | 23 | 0 |
| tiny Float32 | 23 | 0 |
| full Float64 | 35 | 0 |
| full Float32 | 35 | 3 |

The full Float32 failures are the parameter-direction JVP (99 of 26,368
components), `layers.3.weight` VJP (2 components), and `layers.4.weight` VJP
(34 components). The parameter JVP has maximum absolute disagreement
`2.09808349609375e-5` and normalized L2 error about `8.5e-7`.

In the Float64-promotion control, the parameter JVP's maximum absolute error
is approximately `1.72e-5` for PyTorch Float32 and `1.81e-5` for JAX Float32;
both have normalized L2 error about `8.3e-7`. This supports a floating-point
rounding explanation for these quantities. It does not change the original
failed comparisons into passes. The later Float32 tolerance choice is documented
separately in [the plan](../../../plan.md#5-forward-validation-over-a-declared-domain).

These measurements do not include JAX optimizer updates, stochastic dropout,
or the subsequently added domain sweep. Current exporter/adapter commands are
in the [porting guide](../../../src/porting.md#run-the-independent-jax-reference);
regeneration with the current code is a new run, not a byte-identical recreation
of these earlier HDF5 files.
