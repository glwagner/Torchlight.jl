# Torchlight.jl — plan for validated neural-network translation (revised)

Revision 2, 10 September 2026. Repository: `glwagner/Torchlight.jl` (created,
private). This revision replaces the 10 September draft. It records (a) what the
research into Lux.jl, Luximm.jl, Reactant.jl and the cited GitHub work changed,
(b) what has now actually been executed, and (c) what remains planned. Items
marked **Executed** are backed by code and reports in this repository; items
marked **Planned** are not.

The central question is unchanged: **given PyTorch or JAX source code and, when
available, a checkpoint, what is the quickest route to a Julia-usable model with
the required architecture access, composability, and derivative fidelity?**

The three motivating questions are unchanged: (1) architecture diagnosis and
support in Lux and its AD/execution backends; (2) validation over a clearly
specified range of inputs, shapes, modes, and states; (3) input and parameter
derivative agreement, particularly through Lux/Reactant/Enzyme, and
demonstrated training.

## 0. Summary of what changed in this revision

| Draft statement | Revision |
|---|---|
| Lux Dense/Dropout support was "a hypothesis until the chosen versions run" | **Executed.** Lux 1.31.4: `Dense` weight `(out, in)`, bias `(out,)` vector; `Dropout` state `(rng, training)`; `Lux.testmode` disables it; `Lux.f64` converts the parameter tree. Verified in `src/`, `cases/`. |
| Reactant/Enzyme on the target machine untested | **Executed.** Reactant 0.2.285 + Enzyme 0.13.202 compile and differentiate the full model on Apple M5 Max CPU in Float64 and Float32; native Enzyme.jl also works; all three Julia AD paths agree with PyTorch. |
| Luximm fixture reader forces Float32; dtype-preserving adapters "before Float64 claims" | **Executed.** `Torchlight.read_fixture` preserves dtype and rejects dtype/finiteness violations; `map_parameters` wraps `Luximm.Interop.apply_state_dict` and adds coverage/duplicate/shape/dtype checks, evaluating each transform exactly once. |
| HDF5 axis convention "verify with non-square sentinels" | **Executed.** numpy `(2,3)` → Julia `(3,2)` with `a[j+1,i+1] == a_py[i,j]`; dtype and Int64 preserved; `torch.export` records `aten.linear/tanh/dropout` for the MLP. |
| Tolerances `atol=1e-10, rtol=1e-8` (F64), `atol=1e-6, rtol=1e-4` (F32) as starting thresholds | **Calibrated and frozen** (§5). Forward quantities: strict component-wise. Float32 derivative arrays: same numbers, scale-aware (`relative_to_max`); the strict bar's original failure counts are preserved in §5. Float32 Adam update at `|g| ≲ 10³ε` fails and is **retained as a failure**, not loosened (§7). |
| Lux issue #1657 / PR #1658, Reactant PR #2928 states | Rechecked 10 Sep 2026: all still open; PR #1658 draft (last update Feb 2026); PR #2928 (13 commits, head `df6383d`) not merged, "not quite ready for full review". Unchanged implications. |
| M0 "reproduce one existing Luximm vision variant's parity" | **Deferred** (§9). Luximm's *conventions* were reproduced and reused (fixture layout, `apply_state_dict`, mapping triples, axis reversal); running a timm backbone parity requires timm + HF downloads and adds no evidence about the MLP case. Recorded as not executed. |
| JAX as a second source "after the first case passes" | **Executed in parallel** by a collaborating agent: a plain-JAX evaluator (`python/torchlight_ref/jax_reference.py`) recomputes outputs, gradients, VJP/JVP and domain outputs from the PyTorch fixture arrays and writes a sibling fixture; Python tests check PyTorch↔JAX parity and malformed-fixture rejection. JAX training/dropout remain `not_tested`. |
| Project structure | Realized as proposed (§8) plus `python/tests`, `test/review_regressions.jl`, `docs/src/{porting,review_checklist,fixture_schema}.md`, `AGENTS.md`, `CONTRIBUTING.md`. |

## 1. Translation routes and hybrid approaches

Unchanged in substance. The first route — **translate code, load data, validate
against the original executable** — is implemented for the column MLP. A
checkpoint alone does not describe the forward computation: our own reference
applies `tanh` and dropout *functionally* inside `forward`, so a module listing
shows only `nn.Linear` layers; `torch.export` does reveal them
(`aten.tanh`, `aten.dropout(p, False)` in eval).

| Route | Status | Evidence |
|---|---|---|
| Source translation + direct data transfer | **Executed** | `cases/column_mlp/`, reports in `benchmarks/results/` |
| Graph import (Reactant PR #2928 / torchax / StableHLO) | **Planned** | PR unmerged; parameters embedded as constants; no editable Lux tree. Experiment slot: `experiments/graph_import/` |
| Hybrid composition | **Planned** | after graph import is available |

The common adapter interface exists as `Torchlight.LuxAdapter` with
`forward` and `gradients(adapter, backend, loss, x)` for
`:zygote`, `:enzyme`, `:reactant`; imported/compiled routes should implement
the same two functions and report `unsupported` for anything they cannot do.
Constant capture is guarded by sensitivity checks on inputs *and* parameters
(§5).

## 2. First model and scope

Unchanged: paper-inspired column MLP `420 → 512 → 512 → 512 → 512 → 412`,
tanh hidden, linear output, dropout 0.1, 1,214,876 parameters (verified count).
The PyTorch reference (`python/torchlight_ref/column_mlp.py`) is labelled an
independent reconstruction. Tiny diagnostic size `7 → 11 → 5 → 3`, batch 13;
full size batch 64. Synthetic inputs at O(1) scale stand in for standardized
predictors. The FNN library cross-check and any use of ECMWF data remain
**Planned**.

## 3. Architecture diagnosis and support

Unchanged method. **Executed for the MLP:** Lux expresses every operation with
existing layers (`Dense`, `Dropout`); semantics match (§5); Reactant compiles
each declared shape (batch 64 and the tiny batch 13; other batch sizes run
through plain Lux in the domain sweep); Zygote, native Enzyme and
Reactant+Enzyme differentiate it; timings in §9. One custom layer was written
for *diagnosis*, not support: `FixedMaskDropout`, which takes a 0/1 mask from
its state so train-mode comparisons with shared masks are exact.

Issues found and their routing (none required an upstream PR):

- Native Enzyme.jl requires the loss closure to be marked `Const` (it captures
  the model and state), and forward-mode JVP requires
  `set_runtime_activity(Forward)`; both handled in `src/adapters.jl` and
  the runner. Documented as porting knowledge, not a Lux/Enzyme bug.
- LuxLib warns when `Dropout` in test mode sits inside an AD call; harmless for
  eval-mode parity and left visible.

## 4. Shared fixtures and parameter mapping

**Executed.** Schema v1 is documented in `docs/src/fixture_schema.md`. Relative
to Luximm's `/input`, `/state_dict/<key>`, `/output`, it adds `/meta`
attributes (schema version, source/environment versions, declared architecture,
`dense_weight_layout`, loss), `/target`, `/intermediates`, `/probes`,
`/derivatives`, `/training/{sgd,adam}`, `/domain/<case>`, and a
`MANIFEST.json` with SHA-256 per file. The exporter preserves dtype (no
`.float()` cast); Adam's step counter is stored as an integer attribute.

Mapping decisions verified by execution:

- The dense-weight transform is chosen from the fixture's declared
  `dense_weight_layout`, not inferred from the framework name (a plain-JAX
  fixture deliberately keeps `(out, in)`).
- Reusing the value transform for gradient arrays is correct only for index
  permutations/reshapes; `compare_tree` documents that a scaling or nonlinear
  transform needs a separate gradient-coordinate transform.
- 0-D datasets are wrapped as 0-dimensional arrays so integer buffers such as
  `num_batches_tracked` survive in `Dict{String,Array}`.

## 5. Forward validation over a declared domain

**Executed.** Per fixture the runner checks: output; every intermediate stage
(`layer{i}_preact`, `layer{i}_act`) with the expected stage set enforced;
sensitivity to input and to parameter perturbations; and the declared-domain
sweep exported by the source: zeros, ones, constant −2, the feature basis, a
ramp, Gaussian inputs at scales 0.01/1/10/100 and 1e3, batch sizes
1/2/7/32/256, a permuted reference batch, and the two halves of the reference
batch. The required case list is fixed in the runner
(`REQUIRED_DOMAIN_CASES`) so a fixture missing families cannot pass. Column
independence (split and permuted batches) is also checked Julia-internally.

Tolerance policy, calibrated on the four fixtures and frozen for the acceptance
runs (`cases/column_mlp/validate.jl`):

| Quantity | Float64 | Float32 | Form |
|---|---|---|---|
| Forward outputs, intermediates, domain cases, loss | `atol 1e-10, rtol 1e-8` | `atol 1e-6, rtol 1e-4` | strict component-wise |
| Gradients, VJPs, JVPs, update deltas, Adam moments | `atol 1e-10, rtol 1e-8` | `atol 1e-6, rtol 1e-4` | Float32: `relative_to_max` (denominator `max(|e_i|, max|e|)`) |
| Finite-difference directional derivative | relative `1e-6` | relative `1e-2` | best step in a sweep |
| Cross-framework adjoint consistency | relative `1e-8` | relative `1e-4` | |

Every `Evidence` record serializes `atol`, `rtol` and the `relative_to_max`
flag. The change from the draft is the explicit scale-aware form for Float32
derivative arrays. Original strict-bar results on the full Float32 fixture,
preserved as the calibration record (independent JAX evaluator vs PyTorch,
before any tolerance change): parameter JVP failed on 99 of 26,368 compared
components (max-abs 2.1e-5), `layers.3.weight` VJP on 2 components,
`layers.4.weight` VJP on 34 components; normalized L2 of all three 6e-7 to
9e-7. Re-evaluating the same Float32 arrays after promotion to Float64 gave
max-abs deviations of 1.7e-5 (PyTorch F32 vs F64) and 1.8e-5 (JAX F32 vs F64)
with normalized L2 ≈ 8e-7 in both. That two source frameworks differ from each
other by this much *motivates* the chosen scale-aware Float32 derivative bar;
it does not prove that no port could meet the strict bar, and the strict-bar
counts remain the reference for any future tightening.

Measured results (10 Sep 2026, Apple M5 Max, Julia 1.12.7, torch 2.14.0):

| Fixture | Forward max-abs | Intermediates max-abs | Domain (18 families) | Status |
|---|---|---|---|---|
| tiny F64 | 8e-17 | ≤ 1e-16 | all pass | accepted (71 passed, 0 failed) |
| tiny F32 | 3e-8 | ≤ 1e-7 | all pass | accepted (71 passed, 0 failed) |
| full F64 | 6e-16 | 4e-15 | all pass | accepted (75 passed, 0 failed) |
| full F32 | 2.5e-7 | 9.5e-7 | all pass | **not accepted** (Adam, §7): 72 passed, 3 failed (`adam_params_after` 1 component; `adam_update_delta` 102 of 1,214,876 components across the five weight blocks, 15/20/20/18/29; plus the acceptance verdict) |

Injected defects (feature permutation, zeroed bias, wrong loss denominator,
zero input gradient, incomplete mapping) are each flagged by the harness and
recorded as `defect_detection` evidence in every report.

## 6. Gradient fidelity

**Executed** for PyTorch → Lux on all three Julia AD paths:

- MSE loss value, `∂L/∂x`, and every block of `∂L/∂θ` (compared in Lux layout
  after the same index transform used for loading).
- VJPs with the fixture's arbitrary output cotangent `v`: `∂/∂(x,θ) Σ v·f`.
- JVPs from the source (`J_x u_x`, `J_θ u_θ`) used in a **cross-framework
  adjoint check** `⟨v, J u⟩ = ⟨Jᵀv, u⟩` with the Lux VJP, for inputs and
  parameters separately; plus a Lux-side forward-mode JVP through Enzyme.
- Central finite differences in the fixture's input and parameter directions
  with a step sweep.
- Reactant-vs-Zygote agreement, and compiled-function anti-constant checks
  (perturbing `x` and `θ` must change the compiled output).

The diagnostic matrix from the draft is realized as: PyTorch CPU (F64 and
F32 fixtures); Lux+Zygote; Lux+native Enzyme; Lux+Reactant+Enzyme (CPU); JAX
CPU F64/F32 via the collaborating adapter. Accelerator paths remain
**Planned**.

## 7. Training fidelity

**Executed:** one SGD step and one Adam step from identical arrays with
`Optimisers.jl` (`Descent`, `Adam(lr, (β1, β2), ε)`), comparing gradients,
post-update parameters, the update *delta*, and Adam's `exp_avg`/`exp_avg_sq`
against `torch.optim`; the exact update rule is recorded in the fixture. A
compiled `Lux.Training.single_train_step!` with `AutoEnzyme` under Reactant
reproduces the SGD step (optional diagnostic, passed on all fixtures).

Finding: in Float32 the full model's Adam step fails the frozen bar on 102 of
1,214,876 update components spread over all five weight blocks (15, 20, 20, 18,
29 per block; worst block `layers.3.weight`, max-abs 6.9e-6); parameters after
the step fail on 1 component. The
first Adam step is `lr·g/(|g|+ε)`; for `|g| ≲ 10³ε` a Float32 rounding
difference in `g` changes the update at O(1) relative size in either framework.
Three pieces of evidence attribute the failure: (i) the runner's
`adam_update_delta_diagnosis` shows all 102 failing components have
`|g| < 10³ε`; (ii) the Float64 fixture passes the identical check; (iii) an
independent control by the collaborating reviewer fed the exact
PyTorch-exported Float32 gradients into `Optimisers.Adam` and reproduced
PyTorch's parameters-after and update to max-abs 3.7e-9 with zero failures
under the frozen tolerance. Hence the update rules agree at these gradients and
the mismatch is Float32 gradient disagreement amplified by `g/(|g|+ε)`. The
acceptance verdict stays **not accepted** for full F32; the resolution options
(a Float32-specific update budget conditioned on `|g|`, or excluding
ill-conditioned components with an explicit count) are a documented decision
for the next revision, not a silent change.

Dropout: eval-mode fixtures carry no masks; train-mode fixtures carry explicit
0/1 masks and the runner reproduces output, input gradient and parameter
gradients exactly (F64) via `FixedMaskDropout`. Native `Lux.Dropout` keep
fraction and `1/(1−p)` scaling are checked in-framework; matching stochastic
trajectories across frameworks is recorded as `unsupported` by design.
Multi-step deterministic training, checkpoint export from Python followed by
continued training in Lux, and statistical comparison of stochastic training
remain **Planned**.

## 8. Utilities built

Realized layout:

```text
Torchlight.jl/
  src/                 fixture.jl mapping.jl compare.jl report.jl probes.jl adapters.jl
  cases/column_mlp/    ColumnMLP.jl (model, mapping, FixedMaskDropout) validate.jl (runner)
  python/torchlight_ref/  column_mlp.py export.py dump_column_mlp.py jax_reference.py
  python/tests/        PyTorch↔JAX parity and malformed-fixture tests
  test/                runtests.jl review_regressions.jl fixtures/ (tiny, committed)
  benchmarks/          run_all.sh, results/ (uncommitted reports)
  docs/                plan.md src/{index,porting,review_checklist,fixture_schema}.md
  experiments/         graph_import/ (empty; planned)
```

Reuse boundary with Luximm: dependency, pinned to `8e26bbe` via Project
`[sources]`; we call `Interop.apply_state_dict` and follow its fixture layout
and mapping-triple convention; we do not call `read_parity` or `axis_reverse`
because both cast to Float32. A dtype-preserving reader/transform is the
plausible first upstream contribution (**Planned**).

## 9. Milestones, effort, and exit criteria

| Milestone | Draft budget | Outcome |
|---|---|---|
| M0 environment, contract, conventions | 0.5–1 day | **Partial.** Tested versions recorded and now locked (`Manifest.toml` committed; `python/requirements-lock.txt`); HDF5/dtype/axis conventions verified; Reactant+Enzyme probe passed; tiny MLP mapping verified. Luximm vision-variant parity **not run** (deferred; see §0). |
| M1 forward port | 0.5–1.5 days | Done: all four fixtures pass forward, intermediates, domain. |
| M2 derivative fidelity | 1–2 days | Done for F64 and F32 across Zygote/Enzyme/Reactant with FD, adjoint and cross-framework checks. |
| M3 training + report | 1–2 days | **Partial.** One SGD/Adam step, shared-mask dropout, compiled train step, Markdown/JSON reports done; full F32 not accepted (Adam); continuation training **not done**. |
| M4 second source + harder case | 2–5 days | **Partial.** JAX evaluator done (forward/AD only). Convolution/normalization case **Planned**. |

Effort record (one working session, 10 Sep 2026; two AI agents in parallel
with human oversight): the working directory was created at 15:08 local time,
the first commit landed at 15:16, and the first accepted full-Float64 report
was written at about 15:40 — roughly 25 minutes from first commit to an
accepted full-model report, and roughly 40 minutes including plan review and
research. Per-phase timings were **not measured** and are not estimated here;
the review-driven fixes (transform double-evaluation, `relative_to_max`
default, JSON escaping, empty-collection edge cases, mode/loss validation,
backend-subset test semantics) are listed so a future port can count them.
Code size: `src/` ≈ 600 lines, case ≈ 560 lines, Python reference/exporter ≈
450 lines. No upstream changes were needed.

Runtime (full model, batch 64, CPU, from `benchmarks/results/*.json`):

| Step | Float64 | Float32 |
|---|---|---|
| plain Lux forward, warm | 1 ms | < 1 ms |
| Zygote gradient, first / warm | 4.7 s / 3 ms | 4.9 s / 2 ms |
| native Enzyme gradient, first / warm | 0.09 s / 7 ms | 0.09 s / 1 ms |
| Reactant forward compile / first run / warm run | 1.3 s / 29 ms / 1 ms | 1.3 s / 30 ms / 1 ms |
| Reactant+Enzyme gradient compile / first run / warm run | 13.1 s / 0.22 s / 2 ms | 14.1 s / 0.24 s / 2 ms |
| compiled `single_train_step!` first call (incl. compile) | 26 s | 28 s |

All Reactant "run" figures include host-transfer synchronization. Peak memory
and device transfers are **not measured**.

Status against exit criteria: M1 and M2 met within one session versus the ≤ 1
week budget; M0 and M3 partial as tabulated. No backend was removed. The
Float32 derivative bar was made scale-aware *before* the acceptance runs, is
recorded on every evidence record, and the strict-bar failure counts are
preserved in §5.

## 10. Decisions, proposals, and remaining choices

Decided and executed: name Torchlight.jl; owner `glwagner`; private
repository (flip to public is a one-click choice for the owner); MIT license;
Julia 1.12 with the tested `Manifest.toml` committed and Python versions
frozen in `python/requirements-lock.txt` (Project/requirements ranges remain
the compatibility declaration); Luximm as a pinned dependency; fixture schema v1; PyTorch-first
with a parallel plain-JAX evaluator; strict component-wise tolerances for
forward quantities and scale-aware Float32 derivative tolerances.

Open: Float32 Adam acceptance policy (§7); whether to run a Luximm vision
parity as a control; graph-import experiment scope; whether to propose the
dtype-preserving reader upstream to Luximm; whether to coordinate a generic
loader with Lux PR #1658. No upstream issue, comment, or PR has been created.

## 11. Existing GitHub work and contribution opportunities

Rechecked 10 September 2026 via the GitHub API:

| Work | State on 10 Sep 2026 | Implication |
|---|---|---|
| Lux issue #1657 | open, 6 comments, last activity 3 Feb 2026 | Maintainer favours a limited `PytorchLoader` extension for `nn.Linear`/`nn.Conv`/`nn.Sequential`; our mapping + coverage checks are the validated piece such a loader needs. |
| Lux PR #1658 | open, draft, last update 3 Feb 2026 | Not usable yet; coordinate rather than compete. |
| Reactant issue #2065 / PR #2928 | open / open, not merged, head `df6383d`, "not quite ready" | Graph-import route remains an experiment; parameters as constants. |
| SpeedyWeather PR #959 | open, draft | Candidate second scientific case. |
| Luximm.jl `8e26bbe` (30 Aug 2026) | Julia 1.12, Lux ^1.31.4, HDF5 ^0.17 | Reused as dependency; Float32 casts in reader/transforms confirmed by reading the source and by execution. |
| LuxTestUtils v2.3.1 | released 22 Jul 2026 | Test-only dependency here; `test_gradients` not yet used (our checks compare to an external reference, which it does not take). |
| Lux v1.31.4, Reactant v0.2.285 (6 Sep 2026), Enzyme v0.13.202 (8 Sep 2026) | current | Versions used for all evidence. |

## 12. Concrete Luximm integration

Executed as proposed with one change: rather than patching Luximm's Float32
casts, Torchlight owns the reader (`read_fixture`) and transforms
(`reverse_axes`, `to_lux_dense_weight`) and reuses only
`Interop.apply_state_dict`. The first implementation sequence (1) reproduce a
Luximm fixture comparison, (2) tiny asymmetric MLP, (3) Float64 + gradients,
(4) paper-size MLP + training, was executed as (2)–(4); (1) is deferred.

## 13. Coverage of the project discussion

Unchanged mapping of topics to sections, with the added rows: collaborating
agent review findings (all incorporated, listed in §9), and Python-side test
coverage of the exporter/JAX parity.

## 14. Next steps

1. Decide the Float32 Adam policy and re-run `benchmarks/run_all.sh`.
2. Multi-step deterministic training and Python-checkpoint → Lux continuation
   (`/training` schema extension with optimizer state and batch order).
3. Graph-import experiment against Reactant PR #2928 on the tiny fixture,
   reporting supported signatures, constants, and derivative capabilities.
4. Second architecture (convolution or normalization) to stress mapping and
   layer coverage; SpeedyWeather's surface-roughness MLP as a second
   scientific case.
5. Optional: Luximm vision-variant parity as a control; propose a
   dtype-preserving reader upstream.
