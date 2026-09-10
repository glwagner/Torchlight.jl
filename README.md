# Torchlight.jl

**Validated translation of neural-network parameterizations for Earth system
models from PyTorch/JAX into [Lux.jl](https://lux.csail.mit.edu).**

Machine-learned parameterizations (radiation, convection, turbulence, surface
fluxes, sea-ice and ocean closures) are usually developed in PyTorch or JAX.
Using one inside a differentiable Julia Earth system model means translating
the architecture into Lux, loading the trained parameters, and trusting the
result: its outputs, its derivatives with respect to inputs and parameters, and
its behaviour under continued training.

Torchlight is a *validation harness* for that translation, not a converter.
Given a network written in PyTorch or JAX (and, when available, a checkpoint),
it answers three questions with recorded evidence:

1. Can the architecture be expressed natively in Lux and executed/differentiated
   through Lux's backends (Zygote, native Enzyme, Reactant + Enzyme)?
2. Does the translated computation match the source over a *declared* input
   domain, layer by layer, not just on one random batch?
3. Do input and parameter derivatives, and one optimizer step, agree with the
   source framework closely enough to continue training in Julia?

The method is fixture-first: the source framework exports real arrays once
(inputs, parameters, intermediates, derivatives, optimizer updates, a declared
input domain), the Julia side maps every parameter with full coverage checks,
and an evidence report records every capability as `passed`, `failed`,
`unsupported`, or `not_tested`. Injected defects verify that the harness
catches the mistakes it claims to catch.

## Status (10 September 2026)

First case: a paper-inspired atmospheric column MLP
(420 → 512 → 512 → 512 → 512 → 412, tanh, dropout 0.1; 1,214,876 parameters),
after [Farchi et al. 2025](https://doi.org/10.1002/qj.4934), who train a
column-wise network as a neural correction inside an operational weather
model. The PyTorch reference is an independent reconstruction, not the
authors' code.

| Fixture | Result | Notes |
|---|---|---|
| tiny (7→11→5→3) Float64 | accepted | all backends, 71 checks |
| tiny Float32 | accepted | all backends, 71 checks |
| full (paper size) Float64 | accepted | all backends, 75 checks, Reactant compile + run timed |
| full Float32 | **not accepted** | 73 passed; Adam first-step update fails the frozen Float32 tolerance on 102 of 1,214,876 components (all with `|g| < 10³ε`). Diagnosis and the independent optimizer control are in [docs/plan.md §7](docs/plan.md) and [docs/results/2026-09-10](docs/results/2026-09-10/README.md). |

Machine-generated reports for this run are committed under
`docs/results/2026-09-10/` (Markdown and JSON with fixture hashes and tested
versions); `benchmarks/results/` holds uncommitted scratch reruns. See [docs/plan.md](docs/plan.md) for the full plan, decisions, and
findings, and [docs/src/](docs/src/index.md) for the porting guide, review
checklist, and fixture schema.

Scope of the JAX adapter (`python/torchlight_ref/jax_reference.py`): an
*independent plain-JAX evaluator* of the same dense/tanh network that consumes
the PyTorch fixture's arrays and re-derives outputs, gradients, VJPs, JVPs and
domain outputs. It is not a Flax checkpoint converter.

## Layout

```
src/                      Torchlight: fixture reader, parameter mapping, compare, report, probes, adapters
cases/column_mlp/         Lux model, parameter mapping, acceptance runner (validate.jl)
python/torchlight_ref/    PyTorch reference model + fixture exporter; JAX evaluator
python/tests/             Python unit tests (exporter/JAX parity, malformed-fixture rejection)
test/                     Julia tests, including adversarial review regressions; tiny fixtures
benchmarks/               run_all.sh and (uncommitted) results/
docs/                     plan.md, porting guide, review checklist, fixture schema
```

## Quick start

```bash
# Python side (once): reference model + fixtures
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
cd python && ../.venv/bin/python -m torchlight_ref.dump_column_mlp --out ../data/fixtures && cd ..

# Julia side
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cases/column_mlp/validate.jl data/fixtures/column_mlp_tiny_float64.h5 benchmarks/results
benchmarks/run_all.sh            # every eval fixture; nonzero exit if any is not accepted
julia --project=. -e 'using Pkg; Pkg.test()'
PYTHONPATH=python .venv/bin/python -m unittest discover -s python/tests
```

Requires Julia 1.12 and Python ≥ 3.12 with `torch`, `h5py`, `numpy`, `jax`.
All Julia dependencies are registered packages.

To reproduce the committed reports with the tested versions, use the committed
`Manifest.toml` (instantiate as above) and the primary-package version snapshot
`python/requirements-lock.txt` (five primary packages; transitive dependencies
are not locked):

```bash
.venv/bin/pip install -r python/requirements-lock.txt
```

## Using the harness for another model

1. Write the source reference and export a fixture with
   `python/torchlight_ref/export.py` conventions (see
   [fixture schema](docs/src/fixture_schema.md)).
2. Implement the Lux model and a `ParameterMapping` of
   `(source_key, lux_path, transform)` triples; load with
   `Torchlight.map_parameters` (fails on any unused key, unmapped leaf,
   duplicate, shape or dtype mismatch).
3. Compare with `Torchlight.compare` / `compare_tree`, record `Evidence` into a
   `Report`, and gate on `acceptance(report, REQUIRED)`.
4. Follow the [porting guide](docs/src/porting.md) and
   [review checklist](docs/src/review_checklist.md).

## Related work

- [Luximm.jl](https://github.com/csvance/Luximm.jl) ports timm vision
  backbones to Lux with pretrained weights. Its porting guide (export reference
  fixtures, map every state-dict key, compare stage by stage, fail on missing
  keys) inspired the fixture-first workflow here; Torchlight targets Earth
  system parameterizations rather than vision models and shares no code.
- Lux [issue #1657](https://github.com/LuxDL/Lux.jl/issues/1657) /
  [PR #1658](https://github.com/LuxDL/Lux.jl/pull/1658): a proposed generic
  PyTorch weight loader for Lux.
- Reactant [PR #2928](https://github.com/EnzymeAD/Reactant.jl/pull/2928):
  StableHLO import of exported PyTorch graphs (parameters become constants).
- SpeedyWeather [PR #959](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/959):
  a data-driven surface-roughness scheme, a candidate second case.
- [Boltz.jl](https://github.com/LuxDL/Boltz.jl) and
  [LuxTestUtils](https://github.com/LuxDL/Lux.jl/tree/main/lib/LuxTestUtils):
  Lux model collection and gradient-testing utilities.

## License

MIT.
