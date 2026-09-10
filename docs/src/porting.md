# Porting a model with Torchlight

A successful port pairs readable Lux code with reproducible evidence of agreement
with its source. This guide is a workflow for a person or LLM; it does not imply that
Torchlight automatically translates arbitrary Python or supports every operation.
Consult the README and generated reports for implemented and executed capabilities.

## Establish the contract

Record the runnable source revision, checkpoint identity, framework versions, and
license. Inspect the forward function as well as its layer list: reshaping, indexing,
normalization, branches, and custom operations may live outside registered modules.
Distinguish parameters, non-trainable buffers, optimizer state, and random state.

Declare the input/output axes, valid shapes, precision, train/eval modes, loss
definition, and required derivatives. Record preprocessing, feature order, units,
and output scaling. If only a paper is available, label the reference as an
independent reconstruction; synthetic agreement does not reproduce scientific skill.

Write a small inventory before coding:

| Source operation | Lux implementation | Semantic detail to test | Execution evidence |
| --- | --- | --- | --- |
| Dense / linear | `Lux.Dense` | Weight axes, bias shape, activation | Per-layer forward and gradients |
| Reshape / transpose | Julia array operation | Named logical axes and ordering | Asymmetric sentinel |
| Normalization | Matching layer or explicit function | Axes, epsilon, variance, running state | Outputs and state transitions |
| Dropout | Native or diagnostic masked operation | Placement, scaling, mode, RNG state | Conditional and stochastic checks separately |
| Unsupported operation | Minimal custom layer or scoped imported block | Explicit boundary parameters and state | Forward and derivatives at boundary |

Each execution claim needs a shape, dtype, mode, and backend. Expressibility in Lux
alone does not demonstrate compilation or derivative support.

## Export the reference first

Save actual inputs, parameters, state, expected outputs, and the metadata needed to
interpret them. For derivative tests, also save targets or output cotangents and
the source derivatives. A common scalar objective is
`sum(abs2, prediction - target) / length(target)`; record that denominator explicitly.
For a general VJP use `sum(cotangent .* prediction)` with the same cotangent in both
frameworks. Seeds are provenance, not a substitute for shared arrays.

Start with a small non-square model and batch sizes that differ from its feature
counts. Preserve Float64 through export, loading, parameter replacement, and
evaluation when making Float64 claims. Record the effective source dtype; enabling
JAX X64 must happen before creating Float64 arrays.

Torchlight's fixture workflow draws on
[Luximm's porting guide](https://github.com/csvance/Luximm.jl/blob/8e26bbee61d899f278e2c771bb10742a66cf3315/docs/src/porting.md).
Its input/state-dictionary/output convention makes reference capture and layerwise
debugging reusable. Check the installed reader and transforms before reusing them:
the inspected Luximm revision casts some arrays to Float32. A dtype-preserving
extension must be validated independently.

## Separate serialization from mathematical layout

For dense layers, the logical conventions are:

| Array | PyTorch | Common Flax Dense | Lux |
| --- | --- | --- | --- |
| Input batch | `(batch, input)` | `(batch, input)` | `(input, batch)` |
| Weight / kernel | `(output, input)` | `(input, output)` | `(output, input)` |
| Output batch | `(batch, output)` | `(batch, output)` | `(output, batch)` |

These are framework conventions, before any serialization effects. h5py/HDF5.jl
interchange can reverse observed axes. Confirm the actual reader with a non-square
sentinel such as `A[i,j] = 100i + j`; assert selected values and shapes. Then write
the source-to-destination transform for each array. Do not transpose every tensor
using one rule. See the [Lux layer reference](https://lux.csail.mit.edu/stable/api/Lux/layers)
and [Flax migration guide](https://flax.readthedocs.io/en/latest/migrating/pytorch_to_jax_flax.html).

Initialize the Lux model to obtain a parameter/state template, then replace all
declared leaves with reference arrays. Check missing and unexpected source keys,
missing destination leaves, duplicate destinations, shapes, and dtypes. Any
intentional unused buffer needs a reason. Check tied parameters explicitly: copying
one tensor into two independent trainable leaves changes the model's parameterization.
Bind the new tree returned by a functional mapping API.

Gradient coordinates need their own check. For a permutation or reshape, apply
the inverse transformation to bring a Lux gradient back to source coordinates.
For a general parameter transformation, use its chain rule; an inverse value
transformation is not generally the correct gradient transformation.

## Implement and diagnose forward behavior

Prefer ordinary Lux composition with explicit parameters and state. Begin in eval
mode, with stochastic operations disabled. Compare each affine result and activation
when possible, then the full model. Test a physical wrapper separately from its MLP.

Use zero, basis, constant, asymmetric, random, and saturated inputs; include multiple
batch sizes. Test column permutation and split/reassembled batches when the model is
supposed to act independently on columns. For stateful models, compare repeated
calls, returned state, and explicit resets. Define the domain actually exercised.

Require equal shapes before comparing elements. For a reference element `r` and
candidate `c`, a common acceptance rule is
`abs(c - r) <= atol + rtol * abs(r)`. Record the tolerances, worst absolute error,
normalized error, and failing component count. Reject unexpected nonfinite values.
Do not replace componentwise acceptance with a single global norm that can hide
a large error in a small component.

When a comparison fails, inspect the first divergent stage. Verify layout, bias,
activation formula, precision, and mode before attributing the difference to roundoff.
Keep the failing case as a regression fixture. Numerical agreement on a finite
suite establishes evidence over the declared cases, not all possible inputs.

## Establish derivative and training evidence

Compare both input and every parameter-block derivative for the same objective.
Use several shared cotangents so agreement does not depend on one loss direction.
On tiny models, finite differences or full Jacobians provide a tractable additional
check; on large models, use selected directional derivatives and a step-size sweep.
Adjoint consistency is useful alongside independent finite differences.

Run ordinary Lux and the intended compiled backend separately. Perturb explicit
parameters and inputs and confirm that compiled results change; accidentally captured
constants can otherwise appear to work. Report native Enzyme and Reactant/Enzyme
as separate execution paths.

[LuxTestUtils](https://lux.csail.mit.edu/stable/api/Testing_Functionality/LuxTestUtils)
provides Julia derivative diagnostics with a finite-difference reference. It belongs
in a testing environment and supplements source-framework comparisons. It does not
replace a PyTorch/JAX reference or establish that an unexecuted backend works.

For training, start with one deterministic SGD update from identical arrays, then
compare loss, gradients, updated parameters, and post-update outputs. Adam additionally
requires moment state, step count, epsilon placement, bias correction, and weight
decay semantics. Starting a fresh optimizer on imported weights is fine-tuning;
exact resumption requires optimizer state and the same subsequent batch order.

For exact conditional dropout comparisons, export masks and match placement and
inverted-dropout scaling. This tests the masked computation. Native dropout needs
separate mode, probability, reproducibility, and random-state tests.

## Deliver the evidence

Publish a readable implementation, reference-generation commands, mapping, domain,
versions, fixture hashes, and an executed result report. Distinguish passed, failed,
unsupported, and not-tested capabilities, including missing prerequisites. A required
missing case must prevent an acceptance pass. Give expected and executed counts.

Record porting effort and obstacles separately from compile and runtime measurements.
Use the [review checklist](review_checklist.md) before claiming a milestone complete.
Extract a reusable utility only after a concrete port demonstrates its need.

## Run the independent JAX reference

For the included dense/tanh column MLP, the JAX adapter consumes an eval-mode
PyTorch fixture and evaluates the same arrays on CPU. From the repository root,
after installing the Python requirements:

```sh
PYTHONPATH=python .venv/bin/python -m torchlight_ref.dump_column_mlp --sizes tiny --out data/fixtures
PYTHONPATH=python .venv/bin/python -m torchlight_ref.jax_reference \
    --input data/fixtures/column_mlp_tiny_float64.h5 \
    --out data/fixtures/column_mlp_tiny_float64_jax.h5
PYTHONPATH=python .venv/bin/python -m unittest discover -s python/tests -v
```

The sibling fixture contains JAX outputs, intermediates, MSE gradients, cotangent
VJPs, and directional JVPs. It keeps weights and parameter derivatives in the
original `(output, input)` coordinates so the same Julia mapping can read them.
This evaluator uses plain JAX array operations; it does not validate Flax loading
or automatic recovery of arbitrary JAX architectures.

X64 is enabled before array construction, while each fixture retains its declared
Float32 or Float64 dtype. Metadata records JAX's version, CPU execution, the adapter
source hash, and the input fixture hash. The adapter rejects unsupported activations,
losses, train mode, and masked eval fixtures. It does not copy source training
snapshots into JAX evidence; optimizer and stochastic behavior remain untested.

Generating a sibling fixture is not itself an acceptance comparison. Evaluate it
through the Julia harness and inspect every required capability. Large Float32
directional derivatives can require a distinct, justified tolerance budget; retain
failures at the originally declared thresholds when calibrating that budget.
