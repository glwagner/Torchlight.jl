# Torchlight fixture schema v1 reader.
#
# Fixtures are HDF5 files written by `python/torchlight_ref/export.py`.  Arrays
# are stored in the source framework's row-major order, so HDF5.jl returns
# them with axes *reversed* relative to the source logical order.  For the
# column MLP this reversal is exactly what Lux wants for activations
# (`(batch, features)` -> `(features, batch)`), but *not* for dense weights
# (`(out, in)` -> `(in, out)`), which need `to_lux_dense_weight`.
#
# Unlike `Luximm.Interop.read_parity`, this reader preserves the stored dtype
# (Float64 stays Float64; Int64 stays Int64).

const SUPPORTED_SCHEMA_VERSIONS = (1,)

"""
    Fixture

Container for one reference case.  Fields hold arrays in the HDF5-natural
Julia layout (source axes reversed) with their stored element types.

- `meta::Dict{String,Any}`: attributes of `/meta`.
- `input`, `target`, `output`: reference arrays (`(features, batch)` in Julia).
- `state_dict::Dict{String,Array}`: every source parameter/buffer by source key.
- `intermediates::Dict{String,Array}`: named intermediate activations.
- `probes`, `derivatives`, `training`, `domain`: nested `Dict`s mirroring the
  HDF5 groups; attributes of a group are stored under the key `"attrs"`.
- `path`: the file the fixture was read from.
"""
struct Fixture
    path::String
    meta::Dict{String,Any}
    input::Array
    target::Union{Nothing,Array}
    output::Array
    state_dict::Dict{String,Array}
    intermediates::Dict{String,Array}
    probes::Dict{String,Any}
    derivatives::Dict{String,Any}
    training::Dict{String,Any}
    domain::Dict{String,Any}
end

function Base.show(io::IO, fx::Fixture)
    m = fx.meta
    print(io, "Fixture(", get(m, "model_id", "?"), ", ", get(m, "source_framework", "?"),
          " ", get(m, "source_version", "?"), ", dtype=", get(m, "dtype", "?"),
          ", mode=", get(m, "mode", "?"), ", input=", size(fx.input),
          ", ", length(fx.state_dict), " state_dict keys)")
end

_read_attrs(node) = Dict{String,Any}(k => read_attribute(node, k) for k in keys(attrs(node)))

"""Recursively read an HDF5 group into nested `Dict`s, preserving dtypes.
Attributes are stored under the `"attrs"` key when present."""
function _read_group(g::HDF5.Group)
    out = Dict{String,Any}()
    for k in keys(g)
        node = g[k]
        out[k] = node isa HDF5.Group ? _read_group(node) : read(node)
    end
    a = _read_attrs(g)
    isempty(a) || (out["attrs"] = a)
    return out
end

# 0-D datasets (e.g. BatchNorm `num_batches_tracked`) read back as scalars;
# wrap them in 0-dimensional arrays so the state dict stays `Dict{String,Array}`.
_as_array(x::AbstractArray) = x
_as_array(x) = fill(x)
_read_arrays(g::HDF5.Group) = Dict{String,Array}(k => _as_array(read(g[k])) for k in keys(g))

function _check_float_array(name, a, T)
    eltype(a) === T || error("$name eltype $(eltype(a)) != declared $T")
    all(isfinite, a) || error("$name contains non-finite values")
    return nothing
end

# Recursively check every floating array in a nested Dict; attrs and
# non-floating arrays (masks stored as T are floating and are checked) pass.
function _check_float_tree(name, d::AbstractDict, T)
    for (k, v) in d
        k == "attrs" && continue
        if v isa AbstractDict
            _check_float_tree("$name/$k", v, T)
        elseif v isa AbstractArray && eltype(v) <: AbstractFloat
            _check_float_array("$name/$k", v, T)
        elseif v isa AbstractFloat
            (typeof(v) === T && isfinite(v)) || error("$name/$k scalar $(typeof(v)) $v != finite $T")
        end
    end
    return nothing
end

"""
    read_fixture(path) -> Fixture

Read a Torchlight schema-v1 fixture.  Errors on an unsupported schema version
or missing required datasets (`/meta`, `/input`, `/output`, `/state_dict`).
"""
function read_fixture(path::AbstractString)
    isfile(path) || error("fixture not found: $path")
    h5open(path, "r") do f
        haskey(f, "meta") || error("fixture $path has no /meta group (not a Torchlight fixture)")
        meta = _read_attrs(f["meta"])
        v = get(meta, "schema_version", nothing)
        v in SUPPORTED_SCHEMA_VERSIONS ||
            error("unsupported fixture schema_version=$v (supported: $SUPPORTED_SCHEMA_VERSIONS)")
        for req in ("input", "output", "state_dict")
            haskey(f, req) || error("fixture $path is missing required node /$req")
        end
        input = read(f["input"])
        output = read(f["output"])
        target = haskey(f, "target") ? read(f["target"]) : nothing
        sd = _read_arrays(f["state_dict"])
        isempty(sd) && error("fixture $path has an empty /state_dict")
        inter = haskey(f, "intermediates") ? _read_arrays(f["intermediates"]) : Dict{String,Array}()
        probes = haskey(f, "probes") ? _read_group(f["probes"]) : Dict{String,Any}()
        deriv = haskey(f, "derivatives") ? _read_group(f["derivatives"]) : Dict{String,Any}()
        train = haskey(f, "training") ? _read_group(f["training"]) : Dict{String,Any}()
        domain = haskey(f, "domain") ? _read_group(f["domain"]) : Dict{String,Any}()
        # dtype consistency: declared dtype must match stored floating arrays
        declared = get(meta, "dtype", nothing)
        T = declared == "float64" ? Float64 : declared == "float32" ? Float32 :
            error("fixture declares unknown dtype $declared")
        _check_float_array("/input", input, T)
        _check_float_array("/output", output, T)
        target === nothing || _check_float_array("/target", target, T)
        # Floating state must match the declared dtype; integer/bool buffers
        # (e.g. step counters, index buffers) are preserved as stored.
        for (k, a) in sd
            eltype(a) <: AbstractFloat && _check_float_array("/state_dict/$k", a, T)
        end
        for (k, a) in inter
            _check_float_array("/intermediates/$k", a, T)
        end
        _check_float_tree("/probes", probes, T)
        _check_float_tree("/derivatives", deriv, T)
        _check_float_tree("/training", train, T)
        _check_float_tree("/domain", domain, T)
        return Fixture(String(path), meta, input, target, output, sd, inter, probes, deriv, train, domain)
    end
end

"""
    source_layout(fx::Fixture, a::AbstractArray) -> Tuple

Return the size of `a` in the *source framework's* logical axis order
(i.e. reversed relative to the Julia array).  Useful in reports.
"""
source_layout(::Fixture, a::AbstractArray) = reverse(size(a))

"""
    reverse_axes(a) -> Array

Dtype-preserving full axis reversal `(d1, …, dN) -> (dN, …, d1)`.  This is
`Luximm.Interop.axis_reverse` without the `Float32` cast.  Applying it to an
HDF5-natural array restores the source framework's logical axis order.
"""
reverse_axes(a::AbstractArray) = permutedims(a, ntuple(i -> ndims(a) + 1 - i, ndims(a)))
reverse_axes(a::AbstractVector) = copy(a)

"""
    dense_weight_transform(fx::Fixture) -> Function

Transform that maps a dense/linear weight read from `fx` to Lux's `Dense`
convention, chosen from the fixture's *declared* `dense_weight_layout`
attribute rather than inferred from the source framework (a plain-JAX fixture
may well store `(out, in)` while Flax stores `(in, out)`).

- declared `"(out, in)"` (PyTorch `nn.Linear`, Lux `Dense`): the fixture stores
  row-major bytes, HDF5.jl returns `(in, out)`, and a full axis reversal
  restores `(out, in)`.
- declared `"(in, out)"` (Flax `Dense` kernel): the HDF5-natural array is
  already `(out, in)`; the transform is `identity`.

Fixtures without the attribute are rejected: the layout must be declared.
"""
function dense_weight_transform(fx::Fixture)
    layout = get(fx.meta, "dense_weight_layout", nothing)
    layout === nothing && error("fixture does not declare dense_weight_layout")
    layout == "(out, in)" && return to_lux_dense_weight
    layout == "(in, out)" && return identity
    error("unknown dense_weight_layout $layout")
end

"""
    to_lux_dense_weight(w) -> Matrix

Dtype-preserving axis reversal of a 2-D weight stored in `(out, in)` logical
layout: HDF5-natural `(in, out)` -> Lux `(out, in)`.  Prefer
`dense_weight_transform(fx)`, which consults the fixture's declared layout.
"""
to_lux_dense_weight(w::AbstractMatrix) = reverse_axes(w)

"""
    element_type(fx::Fixture) -> Type

Floating-point element type declared by the fixture.
"""
element_type(fx::Fixture) = fx.meta["dtype"] == "float64" ? Float64 : Float32
