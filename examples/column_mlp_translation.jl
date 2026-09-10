# Translate a PyTorch column MLP into Lux, load its weights, and verify it.
#
#     julia --project=. examples/column_mlp_translation.jl [fixture.h5]
#
# Runs from a fresh clone: it uses the tiny fixture committed under
# test/fixtures/ (7 → 11 → 5 → 3, Float64) unless another fixture is given,
# e.g. data/fixtures/column_mlp_full_float64.h5 after running the exporter.
#
# The PyTorch side (python/torchlight_ref/column_mlp.py) is
#
#     class ColumnMLP(nn.Module):
#         def __init__(self, widths, dropout_p=0.1):
#             self.layers = nn.ModuleList([nn.Linear(widths[i], widths[i+1]) ...])
#         def forward(self, x):
#             for i, layer in enumerate(self.layers):
#                 z = layer(x)
#                 if i == last: return z
#                 x = F.dropout(torch.tanh(z), p=self.dropout_p, training=self.training)
#
# and its state_dict has keys layers.{i}.weight (out, in) and layers.{i}.bias (out,).

using Torchlight
using Lux, Random, LinearAlgebra
using Zygote: Zygote

fixture_path = isempty(ARGS) ? joinpath(@__DIR__, "..", "test", "fixtures", "column_mlp_tiny_float64.h5") : ARGS[1]

# 1. Read what PyTorch exported: inputs, parameters, outputs, derivatives.
fx = read_fixture(fixture_path)
widths = Int.(fx.meta["widths"])
T = Torchlight.element_type(fx)
println(fx)

# 2. Write the same architecture in Lux.  Dropout is identity in test mode,
#    exactly as F.dropout(training=False).
L = length(widths) - 1
layers = Pair{Symbol,Any}[]
for i in 1:L
    push!(layers, Symbol("dense_", i - 1) => Dense(widths[i] => widths[i+1], i == L ? identity : tanh))
    i < L && push!(layers, Symbol("dropout_", i - 1) => Dropout(fx.meta["dropout_p"]))
end
model = Chain(; layers...)
ps, st = Lux.setup(Xoshiro(0), model)
ps = T === Float64 ? Lux.f64(ps) : Lux.f32(ps)
st = Lux.testmode(st)

# 3. Load PyTorch's parameters.  nn.Linear.weight and Lux Dense.weight are both
#    (out, in); the HDF5 round trip reverses axes, so the weight transform
#    restores them.  map_parameters fails on any unused key, unmapped leaf,
#    duplicate, or shape/dtype mismatch.
wt = dense_weight_transform(fx)
mapping = Torchlight.ParameterMapping()
for i in 0:L-1
    push!(mapping, ("layers.$i.weight", (Symbol("dense_", i), :weight), wt))
    push!(mapping, ("layers.$i.bias",   (Symbol("dense_", i), :bias),   identity))
end
ps, report = map_parameters(ps, fx.state_dict, mapping)
println(report)

# 4. Same inputs in, same outputs out?  Inputs arrive as (features, batch),
#    which is Lux's convention already.
y, _ = model(fx.input, ps, st)
println(compare(y, fx.output; name = "forward"))

# 5. Same derivatives?  Differentiate the same mean-squared loss with Zygote
#    and compare ∂L/∂x and every block of ∂L/∂θ against PyTorch autograd.
mse(y, t) = sum(abs2, y .- t) / length(y)
loss(x, ps) = mse(first(model(x, ps, st)), fx.target)
gx, gps = Zygote.gradient(loss, fx.input, ps)
println(compare([T(loss(fx.input, ps))], [fx.derivatives["loss"]]; name = "loss"))
println(compare(gx, fx.derivatives["grad_input"]; name = "grad_input"))
for r in compare_tree(gps, fx.derivatives["grad_params"], mapping)
    println(r)
end

# 6. The model is an ordinary Lux Chain: editable, trainable, compilable with
#    Reactant.  cases/column_mlp/validate.jl runs the full acceptance suite
#    (all AD backends, declared-domain sweep, optimizer steps, dropout, defect
#    detection) and writes a report.
