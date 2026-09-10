# Contributing

Torchlight translates neural-network parameterizations for Earth system models
from PyTorch or JAX into Lux. Contributions can include native Lux ports,
source-framework fixtures, validation tools, documentation, and small reproducible
reports of unsupported operations. The useful unit of progress is a port
accompanied by evidence someone else can reproduce.

Start with the [README](README.md), [porting guide](docs/src/porting.md), and
[review checklist](docs/src/review_checklist.md). Agent contributors should also
read [AGENTS.md](AGENTS.md).

## Development

From the repository root, install the Julia dependencies and run the package tests:

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Use the Julia version specified by `Project.toml`. Follow the README's separate
Python setup and fixture-generation commands for cross-framework acceptance checks.
Package unit tests and an executed source-to-Lux comparison provide different
evidence; include the commands and outcomes for both when applicable.

Keep Python reference generation separate from Julia evaluation. Preserve the
fixture's dtype and document any intentional cast. Prefer a tiny fixture that
exposes the problem to a large checkpoint. Large generated artifacts should have
a reproducible download or generation command and a recorded hash.

## A useful contribution

For a new model, include its source identity and license, native implementation,
parameter/state mapping, fixture generator, tested domain, and validation results.
Describe its role in the host Earth system model, including physical units,
feature order, normalization, and output scaling. Synthetic numerical agreement
and scientific performance in the host model require separate evidence.
Explain which features remain unsupported or untested. An inference-only port is
useful when its scope is explicit; it does not establish trainability.

For a harness change, add a test that would fail without the fix. Useful examples
include a swapped feature, missing bias, incorrect loss denominator, mismatched
array shape, or missing derivative. Keep regression tests small enough for CI.

For documentation, check that commands and local links match the repository. Use
an exact revision when discussing an upstream implementation; distinguish source
inspection from independently reproduced results.

## Pull requests and bug reports

Describe the concrete problem, resulting behavior, validation commands, and any
remaining limitations. Include the source and Julia versions, dtype, shape, mode,
backend, tolerances, and smallest reproducer for numerical discrepancies. Identify
the first failing layer when intermediate outputs are available.

Do not include private checkpoints or data without permission to redistribute them.
Retain attribution and applicable notices for adapted upstream code. Route a generic
layer or backend defect upstream when a minimal reproducer identifies its owner;
model-specific implementations and cross-framework evidence belong here.
