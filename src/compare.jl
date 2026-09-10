# Scale-aware array comparison.
#
# Every comparison records the maximum absolute error, the normalized L2 error
# (‖a − e‖₂ / max(‖e‖₂, tiny)), the worst component, and the number of
# component-wise tolerance failures.  A component passes when
#
#     |a − e| ≤ atol + rtol * max(|e|, scale)
#
# where `scale` is optionally the max-abs of the reference array, so
# near-zero components in a large-magnitude array
# are not held to an absolute bar tighter than the array's own roundoff.
# Non-finite values in either array always fail.

"""
    Tolerance(; atol, rtol, relative_to_max = false)

Component-wise tolerance `|a_i − e_i| ≤ atol + rtol * denom_i`.  By default
`denom_i = |e_i|` (strict component-wise).  With `relative_to_max = true`,
`denom_i = max(|e_i|, maximum(abs, e))`, a scale-aware budget appropriate for
derivative arrays whose near-zero components carry the roundoff of the whole
array.  Opting into the scale-aware form is
a deliberate, per-quantity choice and is always serialized with the evidence.
"""
Base.@kwdef struct Tolerance
    atol::Float64
    rtol::Float64
    relative_to_max::Bool = false
end

Tolerance(atol, rtol) = Tolerance(; atol, rtol)

# Starting budgets from the plan (item 5); calibrate per quantity and freeze.
const DEFAULT_TOLERANCES = Dict(
    Float64 => Tolerance(1e-10, 1e-8),
    Float32 => Tolerance(1e-6, 1e-4),
)
default_tolerance(::Type{T}) where {T} = DEFAULT_TOLERANCES[T]

struct ComparisonResult
    name::String
    passed::Bool
    n_total::Int
    n_fail::Int
    max_abs::Float64
    max_rel::Float64
    normalized_l2::Float64
    ref_scale::Float64
    worst_index::Any
    worst_actual::Float64
    worst_expected::Float64
    nonfinite::Bool
    dtype_mismatch::Bool
    shape_actual::Tuple
    shape_expected::Tuple
    tol::Tolerance
end

function Base.show(io::IO, r::ComparisonResult)
    status = r.passed ? "PASS" : "FAIL"
    @printf(io, "%s %s: max_abs=%.3e max_rel=%.3e nl2=%.3e fails=%d/%d", status, r.name,
            r.max_abs, r.max_rel, r.normalized_l2, r.n_fail, r.n_total)
    r.nonfinite && print(io, " NONFINITE")
    r.dtype_mismatch && print(io, " DTYPE")
    r.tol.relative_to_max && print(io, " (relative_to_max)")
    r.shape_actual == r.shape_expected || print(io, " SHAPE ", r.shape_actual, "≠", r.shape_expected)
end

"""
    compare(actual, expected; name = "", tol, atol, rtol, relative_to_max = false,
            allow_dtype_mismatch = false)

Compare two arrays and return a `ComparisonResult`.  Shape mismatch, empty
arrays, element-type mismatch (unless `allow_dtype_mismatch = true`), or any
non-finite value is a failure.  The arithmetic is done in `Float64`.
"""
function compare(actual::AbstractArray, expected::AbstractArray; name::AbstractString = "",
                 tol::Union{Nothing,Tolerance} = nothing, atol = nothing, rtol = nothing,
                 relative_to_max::Union{Nothing,Bool} = nothing, allow_dtype_mismatch::Bool = false)
    if tol === nothing
        base = default_tolerance(eltype(expected) <: AbstractFloat ? eltype(expected) : Float64)
        tol = Tolerance(atol === nothing ? base.atol : atol, rtol === nothing ? base.rtol : rtol,
                        relative_to_max === nothing ? base.relative_to_max : relative_to_max)
    elseif relative_to_max !== nothing || atol !== nothing || rtol !== nothing
        error("pass either `tol` or (`atol`, `rtol`, `relative_to_max`), not both")
    end
    sa, se = size(actual), size(expected)
    dtype_mismatch = eltype(actual) !== eltype(expected)
    if sa != se || isempty(expected)
        return ComparisonResult(String(name), false, length(expected), length(expected), Inf, Inf, Inf,
                                0.0, nothing, NaN, NaN, false, dtype_mismatch, sa, se, tol)
    end
    a = Float64.(actual)
    e = Float64.(expected)
    nonfinite = !(all(isfinite, a) && all(isfinite, e))
    d = abs.(a .- e)
    ref_scale = isempty(e) ? 0.0 : maximum(abs, e)
    denom = tol.relative_to_max ? max.(abs.(e), ref_scale) : abs.(e)
    allowed = tol.atol .+ tol.rtol .* denom
    fails = .!(d .<= allowed)          # NaN compares false -> counted as failure
    n_fail = count(fails)
    max_abs = isempty(d) ? 0.0 : maximum(d)
    rel = d ./ max.(abs.(e), eps(Float64))
    max_rel = isempty(rel) ? 0.0 : maximum(rel)
    nl2 = norm(a .- e) / max(norm(e), floatmin(Float64))
    worst = argmax(d)
    wa, we = a[worst], e[worst]
    widx = worst isa Integer ? (worst,) : Tuple(worst)
    passed = !nonfinite && n_fail == 0 && (allow_dtype_mismatch || !dtype_mismatch)
    return ComparisonResult(String(name), passed, length(e), n_fail, max_abs, max_rel, nl2, ref_scale,
                            widx, wa, we, nonfinite, dtype_mismatch, sa, se, tol)
end

compare(actual::Number, expected::Number; kwargs...) = compare([actual], [expected]; kwargs...)

"""
    compare_tree(actual::NamedTuple, expected::Dict, keymap; kwargs...) -> Vector{ComparisonResult}

Compare Lux parameter-tree leaves against source-keyed reference arrays.
`keymap` is an iterable of `(source_key, lux_path, transform)` triples (the same
`ParameterMapping` used for loading); each reference array is transformed into
Lux layout before comparison so gradients are compared in a common coordinate
system.

**Restriction:** reusing the *value* transform for derivative arrays is only
correct when the transform is an index permutation/reshape (axis reversal,
`permutedims`, `reshape`).  For a transform that scales, combines, or
otherwise changes parameter values, the chain rule applies and a separate
gradient-coordinate transform must be supplied instead.
"""
function compare_tree(actual::NamedTuple, expected::AbstractDict, keymap; prefix = "", kwargs...)
    out = ComparisonResult[]
    for (key, path, transform) in keymap
        haskey(expected, key) || error("reference is missing key $key")
        push!(out, compare(_getleaf(actual, path), transform(expected[key]);
                           name = string(prefix, key), kwargs...))
    end
    return out
end
