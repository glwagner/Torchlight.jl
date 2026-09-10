# Route adapters: a common interface over different execution/AD paths so the
# same fixtures drive every implementation.  An adapter exposes forward
# evaluation and, when supported, gradients.  Unsupported operations are
# reported as `unsupported` with the exact limitation rather than silently
# skipped.

"""
    LuxAdapter(model, ps, st; name)

Native Lux implementation with explicit parameters `ps` and state `st`.
`forward(adapter, x)` returns `(y, st′)`; `gradients(adapter, backend, loss, x)`
returns `(loss_value, grad_x, grad_ps)` for `loss(y)` a scalar function of the
model output, on one of the backends `:zygote`, `:enzyme`, `:reactant`.
"""
struct LuxAdapter{M,P,S}
    name::String
    model::M
    ps::P
    st::S
end
LuxAdapter(model, ps, st; name = "lux") = LuxAdapter(String(name), model, ps, st)

forward(a::LuxAdapter, x) = a.model(x, a.ps, a.st)
forward(a::LuxAdapter, x, ps) = a.model(x, ps, a.st)

const BACKENDS = (:zygote, :enzyme, :reactant)

backend_available(::Val{:zygote}) = true
backend_available(::Val{:enzyme}) = true
backend_available(::Val{:reactant}) = true
backend_available(b::Symbol) = b in BACKENDS && backend_available(Val(b))

function _loss_closure(a::LuxAdapter, loss)
    return (x, ps) -> loss(first(a.model(x, ps, a.st)))
end

"""
    gradients(a::LuxAdapter, backend::Symbol, loss, x) -> (value, grad_x, grad_ps)

Differentiate `loss(model(x, ps, st)[1])` with respect to both `x` and `ps`.
"""
gradients(a::LuxAdapter, backend::Symbol, loss, x) = gradients(a, Val(backend), loss, x)

function gradients(a::LuxAdapter, ::Val{:zygote}, loss, x)
    f = _loss_closure(a, loss)
    val, back = Zygote.pullback(f, x, a.ps)
    gx, gps = back(one(val))
    return val, gx, gps
end

function gradients(a::LuxAdapter, ::Val{:enzyme}, loss, x)
    f = _loss_closure(a, loss)
    # The closure captures model/state; mark it Const so Enzyme does not try to
    # differentiate through the captured data.
    (; val, derivs) = Enzyme.gradient(Enzyme.ReverseWithPrimal, Enzyme.Const(f), x, a.ps)
    return val, derivs[1], derivs[2]
end

"""
Reactant path: move `x`, `ps`, `st` to Reactant arrays, compile the Enzyme
gradient of the loss, and return host arrays.  Compilation time is reported
separately via the returned `timings` when `return_timings = true`.
"""
function gradients(a::LuxAdapter, ::Val{:reactant}, loss, x; return_timings::Bool = false)
    f = _loss_closure(a, loss)
    xr = Reactant.to_rarray(x)
    psr = Reactant.to_rarray(a.ps)
    str = Reactant.to_rarray(a.st)
    model = a.model
    g(x, ps, st) = Enzyme.gradient(Enzyme.ReverseWithPrimal, Enzyme.Const((x, ps) -> loss(first(model(x, ps, st)))), x, ps)
    t0 = time()
    compiled = Reactant.@compile g(xr, psr, str)
    t_compile = time() - t0
    t0 = time()
    (; val, derivs) = compiled(xr, psr, str)
    # Host conversion synchronizes the device work; time it inside the window
    # so `run` measures completion, not dispatch.
    out = (Float64(_to_host(val)), Array(derivs[1]), _to_host(derivs[2]))
    t_run = time() - t0
    # Second call: warm execution (first call can include thunk loading).
    t0 = time()
    r2 = compiled(xr, psr, str)
    _to_host(r2.derivs[2]); Array(r2.derivs[1])
    t_warm = time() - t0
    return return_timings ? (out..., (; compile = t_compile, run = t_run, warm = t_warm)) : out
end

_to_host(x::AbstractArray) = Array(x)
_to_host(x::Reactant.ConcreteRNumber) = Reactant.to_number(x)
_to_host(x::Number) = x
_to_host(nt::NamedTuple) = map(_to_host, nt)
_to_host(x) = x

"""
    compiled_forward(a::LuxAdapter, x) -> (f_compiled, xr, psr, str, t_compile)

Compile the forward pass with Reactant for the shapes of `x`.
"""
function compiled_forward(a::LuxAdapter, x)
    xr = Reactant.to_rarray(x)
    psr = Reactant.to_rarray(a.ps)
    str = Reactant.to_rarray(a.st)
    t0 = time()
    f = Reactant.@compile a.model(xr, psr, str)
    return f, xr, psr, str, time() - t0
end
