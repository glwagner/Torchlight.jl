# Derivative probes that do not depend on a source framework: finite
# differences, adjoint consistency, and sensitivity (anti-constant) checks.

"""
    directional_derivative(f, x, v; h) -> Float64

Central finite difference `(f(x + h v) − f(x − h v)) / 2h` of a scalar
function `f` in direction `v`, in the precision of `x`.
"""
function directional_derivative(f, x, v; h)
    T = eltype(x)
    hT = T(h)
    return Float64((f(x .+ hT .* v) - f(x .- hT .* v)) / (2hT))
end

"""
    finite_difference_check(f, x, v, analytic; steps = ...) -> NamedTuple

Sweep the finite-difference step and report the best agreement between the
central difference in direction `v` and the analytic directional derivative
`analytic = dot(∇f(x), v)`.  Returns `(best_h, best_error, best_relative,
errors::Vector)` where `errors[i]` is `(h, fd, abs_error)`.  Callers decide the
acceptance threshold; the sweep exposes the convergence region before roundoff
dominates.
"""
function finite_difference_check(f, x, v, analytic::Real;
                                 steps = eltype(x) === Float64 ? (1e-3, 1e-4, 1e-5, 1e-6, 1e-7) :
                                                                  (1e-1, 3e-2, 1e-2, 3e-3, 1e-3))
    errors = Tuple{Float64,Float64,Float64}[]
    for h in steps
        fd = directional_derivative(f, x, v; h)
        push!(errors, (Float64(h), fd, abs(fd - Float64(analytic))))
    end
    i = argmin(e[3] for e in errors)
    best_h, best_fd, best_err = errors[i]
    rel = best_err / max(abs(Float64(analytic)), floatmin(Float64))
    return (; best_h, best_fd, best_error = best_err, best_relative = rel, analytic = Float64(analytic), errors)
end

"""
    adjoint_consistency(jvp, vjp) -> NamedTuple

Given `jvp = J*u` (output-shaped) and `vjp = Jᵀ*v` (input-shaped) computed for
probe directions `u`, `v`, check `dot(v, J*u) ≈ dot(Jᵀ*v, u)`.  Returns the two
inner products and their absolute and relative difference.  This alone can
pass for two consistently wrong derivatives; pair with finite differences.
"""
function adjoint_consistency(v, jvp, u, vjp)
    lhs = Float64(dot(vec(v), vec(jvp)))
    rhs = Float64(dot(vec(vjp), vec(u)))
    return (; lhs, rhs, abs_error = abs(lhs - rhs),
            rel_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), floatmin(Float64)))
end

"""
    sensitivity_check(f, x; rng, scale = 1e-2) -> NamedTuple

Perturb `x` and check that `f` changes.  A compiled function that captured its
inputs or parameters as constants returns identical outputs; that must never
pass as a working port.  Returns `(changed::Bool, max_change)`.
"""
function sensitivity_check(f, x; rng = Random.default_rng(), scale = 1e-2)
    y0 = f(x)
    x1 = x .+ eltype(x)(scale) .* randn(rng, eltype(x), size(x))
    y1 = f(x1)
    mc = maximum(abs, Float64.(y1) .- Float64.(y0))
    return (; changed = mc > 0, max_change = mc)
end
