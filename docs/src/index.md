# Torchlight.jl documentation

- [Plan](../plan.md): the research and implementation plan, revised against the
  executed experiments (status, decisions, tolerances, findings).
- [Porting guide](porting.md): the workflow for translating a PyTorch/JAX
  network into Lux with fixture-first validation.
- [Review checklist](review_checklist.md): the evidence a port must present
  before it is called validated.
- [Fixture schema](fixture_schema.md): the HDF5 layout consumed by the Julia
  harness and produced by `python/torchlight_ref`.
- [Results](../results/): committed snapshots of the machine-generated
  acceptance reports (Markdown and JSON, with fixture hashes and tested
  versions), one directory per run date.
- Contributor conventions: [CONTRIBUTING.md](../../CONTRIBUTING.md) and
  [AGENTS.md](../../AGENTS.md).
