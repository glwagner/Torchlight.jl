# Torchlight.jl

**Validated translation of PyTorch/JAX neural networks into [Lux.jl](https://lux.csail.mit.edu).**

Torchlight is a *validation harness*, not a converter. Given a network written
in PyTorch or JAX (and, when available, a checkpoint), it answers three
questions with recorded evidence:

1. Can the architecture be expressed natively in Lux and executed/differentiated
   through Lux's backends (Zygote, native Enzyme, Reactant + Enzyme)?
2. Does the translated computation match the source over a *declared* input
   domain, layer by layer, not just on one random batch?
3. Do input and parameter derivatives, and one optimizer step, agree with the
   source framework closely enough to continue training in Julia?

The approach follows [Luximm.jl](https://github.com/csvance/Luximm.jl)'s
fixture-first porting workflow (export real arrays once, map every state-dict
key, compare intermediates) and extends it with dtype-preserving fixtures,
derivative and optimizer-update groups, declared-domain sweeps, cross-framework
adjoint checks, injected-defect detection, and an evidence report in which every
capability is `passed`, `failed`, `unsupported`, or `not_tested`.

## Status (10 September 2026)

First case: a paper-inspired atmospheric column MLP
(420 → 512 → 512 → 512 → 512 → 412, tanh, dropout 0.1; 1,214,876 parameters),
after [Farchi et al. 2025](https://doi.org/10.1002/qj.4934). The PyTorch
reference is an independent reconstruction, not the authors' code.

| Fixture | Result | Notes |
|---|---|---|
| tiny (7→11→5→3) Float64 | accepted | all backends, 71 checks |
| tiny Float32 | see `benchmarks/results` | |
| full (paper size) Float64 | accepted | all backends, Reactant compile + run timed |
| full Float32 | **not accepted** | Adam first-step update fails the frozen Float32 tolerance on 18 of 262,144 components of one block (max-abs 6.9e-6). The failing components have reference gradients below 10³·ε, which is *consistent with* Float32 ill-conditioning of `g/(|g|+ε)`; the Float64 fixture passes the same check. An independent control (collaborating reviewer) that feeds the exact PyTorch-exported Float32 gradients into `Optimisers.Adam` reproduces PyTorch's parameters and update to 3.7e-9 with zero failures, so the update rule matches and the mismatch comes from Float32 gradient differences amplified by `g/(|g|+ε)`. Retained as a failure under the frozen tolerance, not loosened. |

Machine-generated reports live in `benchmarks/results/` (run locally; not
committed). See [docs/plan.md](docs/plan.md) for the full plan, decisions, and
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

Requires Julia 1.12 (Luximm is pulled from GitHub via `[sources]`, pinned to
commit `8e26bbe`), Python ≥ 3.12 with `torch`, `h5py`, `numpy`, `jax`.

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

Lux [issue #1657](https://github.com/LuxDL/Lux.jl/issues/1657) /
[PR #1658](https://github.com/LuxDL/Lux.jl/pull/1658) (PyTorch loader
proposal), Reactant [PR #2928](https://github.com/EnzymeAD/Reactant.jl/pull/2928)
(StableHLO import of exported PyTorch graphs), Luximm.jl, Boltz.jl,
LuxTestUtils, SpeedyWeather [PR #959](https://github.com/SpeedyWeather/SpeedyWeather.jl/pull/959).

## License

MIT. Luximm.jl (Apache-2.0) is used as a dependency, not vendored.
