"""
    Torchlight

Validated translation of PyTorch/JAX neural networks into Lux.jl.

The package is a *validation harness*, not a converter: it reads reference
fixtures exported by the source framework (see `python/torchlight_ref`), maps
source parameters onto a Lux parameter tree with full coverage checks, and
compares outputs, intermediate activations, input/parameter derivatives, and
optimizer updates across several Julia execution/AD paths.  Every check is
recorded as evidence with a status of `passed`, `failed`, `unsupported`, or
`not_tested`, and rendered into a machine-readable and a Markdown report.
"""
module Torchlight

using HDF5
using LinearAlgebra
using Lux
using Luximm: Luximm
using Optimisers
using Printf
using Random
using Statistics
using Zygote: Zygote
using Enzyme: Enzyme
using Reactant: Reactant

export Fixture, read_fixture, reverse_axes, to_lux_dense_weight, dense_weight_transform, source_layout, element_type
export map_parameters, MappingReport, ParameterMapping
export compare, ComparisonResult, Tolerance
export CapabilityStatus, passed, failed, unsupported, not_tested
export Evidence, Report, record!, write_markdown, write_json, summarize, no_failures, acceptance, compare_tree, default_tolerance
export directional_derivative, finite_difference_check, adjoint_consistency, sensitivity_check
export LuxAdapter, forward, gradients, backend_available, compiled_forward

include("fixture.jl")
include("mapping.jl")
include("compare.jl")
include("report.jl")
include("probes.jl")
include("adapters.jl")

end # module
