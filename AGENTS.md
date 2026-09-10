# Working on Torchlight.jl

Torchlight helps people and LLMs translate neural-network parameterizations for
Earth system models from PyTorch or JAX into editable Lux implementations and
establish what has actually been validated. Read `README.md` and
`docs/src/porting.md` before adding a port. Use `docs/src/review_checklist.md` when
reviewing one.

## Workflow

1. Inspect the executable source, including functional operations, preprocessing,
   state updates, and loss reduction. Record the source revision and environment.
2. Define the case's input shapes, dtypes, modes, and required capabilities before
   implementation. Record physical units, feature order, normalization, and the
   parameterization's role in its host Earth system model. Separate a paper-inspired
   reconstruction from an original model.
3. Export actual reference arrays from the source framework. Matching random seeds
   across languages is insufficient. Start with an asymmetric, small example.
4. Implement ordinary Lux layers with explicit parameters and state. Keep Python
   in fixture generation; the native model must run without a Python interpreter.
5. Check complete parameter/state mapping, logical axes, dtype, and shape before
   comparing numerical outputs. Bind the returned tree from functional mappers.
6. Validate forward values, input derivatives, parameter derivatives, and training
   separately. Report backend, precision, mode, tolerance, and executed case count.
7. Preserve a regression case for each repaired semantic error. Include a deliberate
   defect when extending the harness so its ability to reject errors is exercised.

## Evidence and changes

- Never report an unavailable or unexecuted backend as passing. Required missing
  fixtures are failures; optional omissions need an explicit reason.
- Do not loosen tolerances to make a failing port pass without a numerical
  explanation and a recorded change to the acceptance criteria.
- Reject unexpected nonfinite values, broadcastable shape mismatches, and silently
  truncated comparisons. A finite-domain comparison cannot certify NaN/Inf behavior.
- Keep imported buffers distinct from trainable parameters. Do not silently discard
  unused source keys, unmapped destination leaves, or parameter aliases.
- Read the installed API and upstream source before adding conversion machinery.
  Preserve licenses and notices when adapting code; document reuse decisions.
- Run relevant tests and reproducible examples after changes. Record what ran and
  what remains untested; keep README and compatibility claims consistent with that
  evidence. Do not commit generated environments, credentials, or large downloads.
- When several agents share a checkout, agree on file ownership before editing,
  communicate interface changes, and preserve others' uncommitted work.
