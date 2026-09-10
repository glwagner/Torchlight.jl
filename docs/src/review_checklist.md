# Reviewing a port

Use this checklist to review implementation and evidence together. It is a review
template, not a statement that the current release has completed every item. Mark
inapplicable items with a reason and retain untested capabilities in the report.

## Source and scope

- [ ] Source revision, framework/library versions, license, and checkpoint identity are recorded.
- [ ] Original model, paper-inspired reconstruction, and synthetic fixture are labeled correctly.
- [ ] Functional operations and preprocessing are covered by the architecture inventory.
- [ ] Shapes, axes, feature order, dtypes, modes, and tested input domain are explicit.
- [ ] Each claimed capability identifies its backend and executed cases.

## Data and mapping

- [ ] The fixture generator runs the source implementation and saves actual arrays.
- [ ] A non-square sentinel verifies serialization and framework layout separately.
- [ ] The loader preserves the declared precision and checks destination shapes.
- [ ] Every required source key and destination leaf is accounted for.
- [ ] Duplicate destinations, aliases/tied weights, and intentionally omitted state are handled explicitly.
- [ ] Gradients are compared in the same parameter coordinates with the appropriate chain rule.
- [ ] Native Julia evaluation runs without Python.

## Numerical evidence

- [ ] Outputs match across the declared input families and batch sizes.
- [ ] A failure can be localized to intermediate stages or a small reproducer.
- [ ] Shape mismatch, unexpected nonfinite values, and incomplete comparisons cannot pass.
- [ ] Componentwise tolerances are recorded with error metrics and were fixed before acceptance.
- [ ] Input and all mapped parameter derivatives are checked separately.
- [ ] Loss reduction and shared cotangents are identical across implementations.
- [ ] Independent derivative diagnostics accompany cross-framework agreement.
- [ ] Each compiled signature is exercised with changed inputs and parameters.
- [ ] Train/eval behavior, state transitions, and stochastic behavior are checked when claimed.

## Training and reporting

- [ ] A training claim includes an actual gradient-based update and post-update evaluation.
- [ ] Optimizer-update parity includes the required optimizer state and conventions.
- [ ] Fine-tuning and exact training continuation are distinguished.
- [ ] Expected, executed, failed, and omitted cases are counted; omissions have reasons.
- [ ] Missing required fixtures or backends prevent an acceptance pass.
- [ ] Fixture hashes, generation commands, environments, and reports permit reproduction.
- [ ] README and compatibility statements agree with the saved evidence.

## Test the reviewer’s instruments

When adding harness behavior, deliberately inject an error and verify that the
relevant check fails for the intended reason. Good controls include:

| Defect | Expected diagnostic |
| --- | --- |
| Swap two input features | Sentinel or forward mismatch |
| Drop a bias mapping | Incomplete destination coverage |
| Send a transposed non-square weight | Shape or indexed-value mismatch |
| Use a batch-only MSE denominator | Loss and gradient mismatch |
| Return zero instead of an input derivative | Input derivative mismatch |
| Supply matching infinities | Nonfinite rejection in a finite-domain comparison |
| Remove a required fixture | Acceptance failure, not a successful skip |
| Freeze imported parameters | Parameter perturbation or capability check failure |

For a report that supports only inference, leave derivative and training capabilities
explicitly untested or unsupported. Do not infer them from forward agreement.
