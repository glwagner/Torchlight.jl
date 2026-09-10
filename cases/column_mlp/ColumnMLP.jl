# Native Lux implementation of the paper-inspired column MLP and its parameter
# mapping from the PyTorch reference in python/torchlight_ref/column_mlp.py.
#
# Source architecture (Farchi et al. 2025, §3): widths 420 → 512 → 512 → 512 →
# 512 → 412, tanh after every hidden affine map, linear output, dropout p=0.1
# after each hidden activation during training.  Our PyTorch reference stores
# `layers.{i}.weight` (out, in) and `layers.{i}.bias` (out,).

module ColumnMLPCase

using Lux
using Random
using Torchlight

export column_mlp, column_mlp_mapping, column_mlp_from_fixture, FixedMaskDropout, mse_mean

"""
    column_mlp(widths; dropout_p = 0.1, masked = false)

Lux `Chain` with named layers `dense_i` (and `dropout_i` after each hidden
activation).  With `masked = true` the dropout layers are `FixedMaskDropout`,
a diagnostic layer that takes its 0/1 mask from the layer state so that a
train-mode comparison against the source can use identical masks.
"""
function column_mlp(widths; dropout_p = 0.1, masked::Bool = false)
    L = length(widths) - 1
    L ≥ 1 || error("need at least one layer")
    0 ≤ dropout_p < 1 || error("dropout_p must satisfy 0 ≤ p < 1, got $dropout_p")
    layers = Pair{Symbol,Any}[]
    for i in 1:L
        act = i == L ? identity : tanh
        push!(layers, Symbol("dense_", i - 1) => Dense(widths[i] => widths[i + 1], act))
        if i < L && dropout_p > 0
            drop = masked ? FixedMaskDropout(dropout_p) : Dropout(dropout_p)
            push!(layers, Symbol("dropout_", i - 1) => drop)
        end
    end
    return Chain(; layers...)
end

"""
    FixedMaskDropout(p)

Diagnostic dropout that applies `x .* mask ./ (1 - p)` with `mask` supplied in
the layer state (`st.mask`) when `st.training` is `Val(true)`, and is the
identity in test mode.  It exists to make conditional (mask-given) comparisons
with the source exact; it says nothing about native RNG handling.
"""
struct FixedMaskDropout <: Lux.AbstractLuxLayer
    p::Float64
    function FixedMaskDropout(p)
        0 ≤ p < 1 || error("FixedMaskDropout: p must satisfy 0 ≤ p < 1, got $p")
        return new(Float64(p))
    end
end

Lux.initialparameters(::AbstractRNG, ::FixedMaskDropout) = NamedTuple()
Lux.initialstates(::AbstractRNG, ::FixedMaskDropout) = (; mask = nothing, training = Val(true))

function (d::FixedMaskDropout)(x, ps, st::NamedTuple)
    if st.training isa Val{true}
        st.mask === nothing && error("FixedMaskDropout in training mode requires st.mask")
        size(st.mask) == size(x) || error("mask size $(size(st.mask)) != input size $(size(x))")
        T = eltype(x)
        return x .* st.mask ./ (one(T) - T(d.p)), st
    else
        return x, st
    end
end

"""
    column_mlp_mapping(fx::Torchlight.Fixture) -> ParameterMapping

Map `layers.{i}.weight|bias` onto `dense_i.weight|bias`.  The dense-weight
transform is chosen from the fixture's declared `dense_weight_layout`; biases
are 1-D and need no transform.
"""
function column_mlp_mapping(fx::Torchlight.Fixture)
    wt = Torchlight.dense_weight_transform(fx)
    n = Int(fx.meta["n_layers"])
    keys_ = String.(fx.meta["layer_keys"])
    length(keys_) == n || error("layer_keys length $(length(keys_)) != n_layers $n")
    mapping = Torchlight.ParameterMapping()
    for (i, k) in enumerate(keys_)
        push!(mapping, ("$k.weight", (Symbol("dense_", i - 1), :weight), wt))
        push!(mapping, ("$k.bias", (Symbol("dense_", i - 1), :bias), identity))
    end
    return mapping
end

"""
    column_mlp_from_fixture(fx; masked = false, rng = Xoshiro(0)) -> (model, ps, st, mapping, report)

Build the Lux model declared by the fixture metadata, set it up in the
fixture's element type, and load the fixture's parameters with full coverage
checks.  `st` is returned in test mode for `mode = "eval"` fixtures and in
train mode (with masks installed when `masked = true`) for `mode = "train"`.
"""
function column_mlp_from_fixture(fx::Torchlight.Fixture; masked::Bool = false, rng = Xoshiro(0))
    widths = Int.(fx.meta["widths"])
    fx.meta["activation"] == "tanh" || error("unsupported activation $(fx.meta["activation"])")
    fx.meta["output_activation"] == "identity" || error("unsupported output activation")
    fx.meta["mode"] in ("eval", "train") || error("unknown fixture mode $(fx.meta["mode"])")
    get(fx.meta, "loss", nothing) == "mse_mean" || error("unsupported loss $(get(fx.meta, "loss", nothing)); this case implements mse_mean")
    fx.meta["input_layout"] == "(batch, features)" || error("unsupported input_layout $(fx.meta["input_layout"])")
    size(fx.input, 1) == widths[1] || error("input features $(size(fx.input, 1)) != widths[1] $(widths[1])")
    size(fx.output, 1) == widths[end] || error("output features $(size(fx.output, 1)) != widths[end]")
    fx.target === nothing || size(fx.target) == size(fx.output) || error("target size $(size(fx.target)) != output size $(size(fx.output))")
    p = Float64(fx.meta["dropout_p"])
    model = column_mlp(widths; dropout_p = p, masked)
    ps, st = Lux.setup(rng, model)
    T = Torchlight.element_type(fx)
    ps = T === Float64 ? Lux.f64(ps) : Lux.f32(ps)
    mapping = column_mlp_mapping(fx)
    ps, report = Torchlight.map_parameters(ps, fx.state_dict, mapping; strict = true)
    if fx.meta["mode"] == "eval"
        st = Lux.testmode(st)
    elseif masked
        haskey(fx.probes, "dropout_masks") || error("train-mode masked comparison needs /probes/dropout_masks")
        st = _install_masks(st, fx.probes["dropout_masks"], length(widths) - 2)
    end
    return model, ps, st, mapping, report
end

function _install_masks(st::NamedTuple, masks::AbstractDict, n_hidden::Int)
    expected = Set("layer$i" for i in 0:n_hidden-1)
    Set(k for k in keys(masks) if k != "attrs") == expected ||
        error("dropout mask keys $(collect(keys(masks))) != expected $(collect(expected))")
    for k in expected
        m = masks[k]
        all(v -> v == 0 || v == 1, m) || error("mask $k is not binary")
    end
    pairs = Pair{Symbol,Any}[]
    for k in keys(st)
        s = getfield(st, k)
        ks = string(k)
        if startswith(ks, "dropout_")
            i = ks[length("dropout_") + 1:end]
            haskey(masks, "layer$i") || error("fixture lacks dropout mask layer$i")
            push!(pairs, k => (; mask = masks["layer$i"], training = Val(true)))
        else
            push!(pairs, k => s)
        end
    end
    return (; pairs...)
end

"""Mean over all elements of the squared error, matching the source `mse_mean`.
Shapes must match exactly; broadcasting a mis-shaped target is a silent bug."""
function mse_mean(y, target)
    size(y) == size(target) || error("mse_mean: size(y)=$(size(y)) != size(target)=$(size(target))")
    return sum(abs2, y .- target) / length(y)
end

end # module
