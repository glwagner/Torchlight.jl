# Parameter mapping with coverage checks.
#
# A mapping is a table of `(source_key, destination_path, transform)` triples.
# `map_parameters` rebuilds the Lux parameter tree functionally and enforces
# what a parity test of a trained network needs:
#   * every source key is consumed (source coverage)
#   * every destination leaf is written exactly once (destination coverage,
#     no duplicates)
#   * the transformed array has the shape and element type of the template leaf
#   * each transform is evaluated exactly once, on the validated array

"""
    ParameterMapping

A vector of `(source_key, destination_path, transform)` triples.
`destination_path` is a tuple of
`Symbol`s into the Lux parameter `NamedTuple`; `transform` is applied to the
HDF5-natural array before insertion.
"""
const ParameterMapping = Vector{Tuple{String,Tuple{Vararg{Symbol}},Function}}

struct MappingReport
    n_source_keys::Int
    n_destination_leaves::Int
    n_mapped::Int
    unused_source_keys::Vector{String}
    unmapped_destinations::Vector{String}
    duplicate_destinations::Vector{String}
    shape_mismatches::Vector{String}
    dtype_mismatches::Vector{String}
end

function Base.show(io::IO, r::MappingReport)
    print(io, "MappingReport(", r.n_mapped, "/", r.n_source_keys, " source keys -> ",
          r.n_destination_leaves, " leaves; unused=", length(r.unused_source_keys),
          ", unmapped=", length(r.unmapped_destinations),
          ", duplicates=", length(r.duplicate_destinations),
          ", shape_mismatch=", length(r.shape_mismatches),
          ", dtype_mismatch=", length(r.dtype_mismatches), ")")
end

"""`true` when the mapping is complete and every leaf matched the template."""
iscomplete(r::MappingReport) = isempty(r.unused_source_keys) && isempty(r.unmapped_destinations) &&
                               isempty(r.duplicate_destinations) && isempty(r.shape_mismatches) &&
                               isempty(r.dtype_mismatches)

"""Collect `(path, leaf)` pairs for every array leaf in a parameter tree."""
function _leaves(nt::NamedTuple, prefix::Tuple = ())
    out = Vector{Pair{Tuple,Any}}()
    for k in keys(nt)
        v = getfield(nt, k)
        if v isa NamedTuple
            append!(out, _leaves(v, (prefix..., k)))
        elseif v isa AbstractArray
            push!(out, (prefix..., k) => v)
        end
    end
    return out
end
_leaves(x, prefix::Tuple = ()) = Pair{Tuple,Any}[]

_pathstr(p::Tuple) = join(string.(p), ".")

"""Return a copy of `nt` with the leaf at `path` replaced by `leaf` (non-mutating)."""
function _set_leaf(nt::NamedTuple, path::Tuple, leaf)
    head = first(path)
    haskey(nt, head) || error("leaf path missing key: $head (have: $(propertynames(nt)))")
    if length(path) == 1
        return merge(nt, NamedTuple{(head,)}((leaf,)))
    else
        return merge(nt, NamedTuple{(head,)}((_set_leaf(getfield(nt, head), Base.tail(path), leaf),)))
    end
end

function _getleaf(nt, path::Tuple)
    x = nt
    for k in path
        x = getfield(x, k)
    end
    return x
end

"""
    map_parameters(ps, state_dict, mapping; strict = true) -> (ps′, report::MappingReport)

Apply `mapping` to `ps` after checking coverage, duplicates, shapes and
element types against the template `ps`.

With `strict = true` (default) any incompleteness or mismatch throws; the
report is still available in the error message.  With `strict = false` the
mapped tree and the report are returned and the caller decides.

`ps` is the template returned by `Lux.setup` (converted with `Lux.f64` when the
fixture is Float64).  Transforms must produce arrays of the template leaf's
exact size and element type; the mapper never casts silently.
"""
function map_parameters(ps::NamedTuple, state_dict::AbstractDict{String}, mapping; strict::Bool = true)
    for (k, v) in state_dict
        k == "attrs" && continue
        v isa AbstractArray || error("source entry $k is not an array (got $(typeof(v)))")
    end
    leaves = _leaves(ps)
    leafpaths = Set(_pathstr(p) for (p, _) in leaves)
    used = Set{String}()
    dests = String[]
    shape_mm = String[]
    dtype_mm = String[]
    unknown_dest = String[]
    transformed = Dict{String,Any}()   # transforms are evaluated exactly once
    for (key, path, transform) in mapping
        haskey(state_dict, key) || error("mapping refers to missing source key: $key")
        push!(used, key)
        ds = _pathstr(path)
        push!(dests, ds)
        if ds ∉ leafpaths
            push!(unknown_dest, ds)
            continue
        end
        template = _getleaf(ps, path)
        arr = transform(state_dict[key])
        arr isa AbstractArray || error("transform for $key -> $ds returned a $(typeof(arr)); transforms must return arrays")
        transformed[ds] = arr
        size(arr) == size(template) ||
            push!(shape_mm, "$key -> $ds: got $(size(arr)), template $(size(template))")
        eltype(arr) === eltype(template) ||
            push!(dtype_mm, "$key -> $ds: got $(eltype(arr)), template $(eltype(template))")
    end
    dup = unique([d for d in dests if count(==(d), dests) > 1])
    unused = sort([k for k in keys(state_dict) if k ∉ used && k != "attrs"])
    unmapped = sort([l for l in leafpaths if l ∉ dests])
    isempty(unknown_dest) || error("mapping targets leaves that do not exist in template: $unknown_dest")
    report = MappingReport(length(state_dict), length(leaves), length(used), unused, unmapped,
                           dup, shape_mm, dtype_mm)
    if strict && !iscomplete(report)
        error("incomplete or inconsistent parameter mapping: $report\n" *
              "  unused source keys: $(report.unused_source_keys)\n" *
              "  unmapped destinations: $(report.unmapped_destinations)\n" *
              "  duplicates: $(report.duplicate_destinations)\n" *
              "  shape mismatches: $(report.shape_mismatches)\n" *
              "  dtype mismatches: $(report.dtype_mismatches)")
    end
    # Insert the *validated* arrays (no second transform call).
    ps′ = ps
    for (key, path, _) in mapping
        ps′ = _set_leaf(ps′, path, transformed[_pathstr(path)])
    end
    return ps′, report
end
