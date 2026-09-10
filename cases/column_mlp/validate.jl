# Acceptance runner for the column MLP case.
#
#   julia --project=. cases/column_mlp/validate.jl data/fixtures/column_mlp_tiny_float64.h5 [outdir]
#
# Consumes one eval-mode fixture (and, when present alongside it, the matching
# `_train.h5` fixture) and produces a Report with sections: environment,
# mapping, forward, domain, derivatives (per backend), probes, training,
# dropout, defect_detection, timings.  Acceptance requires every capability in
# REQUIRED to be `passed` and no `failed` evidence anywhere.  Nothing is
# skipped silently: an unavailable backend is `unsupported` with the error, an
# unexecuted check is `not_tested`.

module ColumnMLPValidate

using Lux
using LinearAlgebra
using Optimisers
using Random
using Statistics
using Torchlight
using Enzyme: Enzyme
using Reactant: Reactant
using Zygote: Zygote

include(joinpath(@__DIR__, "ColumnMLP.jl"))
using .ColumnMLPCase

export validate_column_mlp, REQUIRED

const DERIVATIVE_BACKENDS = (:zygote, :enzyme, :reactant)

# Tolerance policy (frozen for the acceptance run; changes must be recorded in
# docs/plan.md §5).  Forward quantities use the strict component-wise bar.
# Float32 derivative arrays use the scale-aware bar (relative_to_max): their
# near-zero components carry the rounding of the whole reduction, which the
# JAX-vs-PyTorch Float32 control also exhibits (~1e-5 max-abs at 1.2M params).
forward_tolerance(::Type{Float64}) = Tolerance(1e-10, 1e-8, false)
forward_tolerance(::Type{Float32}) = Tolerance(1e-6, 1e-4, false)
derivative_tolerance(::Type{Float64}) = Tolerance(1e-10, 1e-8, false)
derivative_tolerance(::Type{Float32}) = Tolerance(1e-6, 1e-4, true)
# Optimizer updates in Float32: one step of lr*grad on O(0.1) weights; the
# update itself is ~1e-4..1e-6 so compare against the *delta* (see training).
update_tolerance(::Type{Float64}) = Tolerance(1e-10, 1e-8, false)
update_tolerance(::Type{Float32}) = Tolerance(1e-6, 1e-4, true)

# Input families the exporter must supply (python/torchlight_ref/export.py::domain_cases).
const REQUIRED_DOMAIN_CASES = ["zeros_b1", "ones_b1", "constant_minus2_b4", "basis", "ramp_b3",
    "randn_scale0.01_b8", "randn_scale1_b8", "randn_scale10_b8", "randn_scale100_b8",
    "randn_b1", "randn_b2", "randn_b7", "randn_b32", "randn_b256",
    "permuted_reference_batch", "reference_batch_first_half", "reference_batch_second_half", "large_1e3_b2"]

const REQUIRED = [
    ("mapping", "coverage"),
    ("forward", "output"),
    ("forward", "intermediates"),
    ("forward", "sensitivity_input"),
    ("forward", "sensitivity_parameters"),
    ("domain", "all_cases"),
    ("domain", "column_independence"),
    ("derivatives/zygote", "mse_loss"),
    ("derivatives/zygote", "grad_input"),
    ("derivatives/zygote", "grad_params"),
    ("derivatives/zygote", "vjp_input"),
    ("derivatives/zygote", "vjp_params"),
    ("derivatives/reactant", "mse_loss"),
    ("derivatives/reactant", "grad_input"),
    ("derivatives/reactant", "grad_params"),
    ("derivatives/reactant", "vjp_input"),
    ("derivatives/reactant", "vjp_params"),
    ("derivatives/reactant", "compiled_forward"),
    ("probes", "finite_difference_input"),
    ("probes", "finite_difference_params"),
    ("probes", "adjoint_consistency_cross"),
    ("training", "sgd_params_after"),
    ("training", "adam_params_after"),
    ("training", "adam_state"),
    ("defect_detection", "permuted_features"),
    ("defect_detection", "missing_bias"),
    ("defect_detection", "wrong_loss_denominator"),
    ("defect_detection", "detached_input_gradient"),
]

_envinfo(fx) = Dict{String,Any}(
    "julia" => string(VERSION),
    "Lux" => string(pkgversion(Lux)), "Reactant" => string(pkgversion(Reactant)),
    "Enzyme" => string(pkgversion(Enzyme)), "Zygote" => string(pkgversion(Zygote)),
    "Optimisers" => string(pkgversion(Optimisers)),
    "fixture" => basename(fx.path),
    "source_framework" => fx.meta["source_framework"], "source_version" => fx.meta["source_version"],
    "source_platform" => get(fx.meta, "platform", ""),
    "dtype" => fx.meta["dtype"], "widths" => string(Int.(fx.meta["widths"])),
    "n_parameters" => sum(length, values(fx.state_dict)),
    "blas_threads" => LinearAlgebra.BLAS.get_num_threads(),
)

"""Compute per-layer pre-activations and activations exactly as the source names them."""
function layer_intermediates(ps, x, n_layers)
    out = Dict{String,Any}()
    h = x
    for i in 0:n_layers-1
        p = getfield(ps, Symbol("dense_", i))
        z = p.weight * h .+ p.bias
        out["layer$(i)_preact"] = z
        if i < n_layers - 1
            h = tanh.(z)
            out["layer$(i)_act"] = h
        end
    end
    return out
end

# Source-keyed reference tree -> Lux-layout comparison of a parameter NamedTuple.
function _compare_params(rep, section, name, ps_like, ref::AbstractDict, mapping, tol; detail = "")
    results = compare_tree(ps_like, ref, mapping; tol)
    ok = all(r -> r.passed, results)
    worst = results[argmax([r.max_abs for r in results])]
    m = Dict{String,Any}("max_abs" => maximum(r.max_abs for r in results),
                         "normalized_l2" => maximum(r.normalized_l2 for r in results),
                         "n_fail" => sum(r.n_fail for r in results), "n_total" => sum(r.n_total for r in results),
                         "atol" => tol.atol, "rtol" => tol.rtol, "relative_to_max" => tol.relative_to_max,
                         "worst_block" => worst.name,
                         "per_block" => Dict(r.name => Dict("max_abs" => r.max_abs, "n_fail" => r.n_fail,
                                                            "normalized_l2" => r.normalized_l2) for r in results))
    record!(rep, Evidence(section, name, ok ? passed : failed,
                          ok ? detail : string(detail, " worst block ", worst.name, ": ", worst), m))
    return ok
end

# Run `f`; on an exception record evidence and return `nothing`.  The default
# status is `failed`: an exception during a required capability is a failure
# of the port until diagnosed.  Optional diagnostics pass `on_error =
# unsupported` explicitly, and the detail records that the classification is
# provisional (the cause may be the harness, not the backend).
function _try(f, rep, section, name; detail = "", on_error::CapabilityStatus = failed)
    try
        return f()
    catch err
        msg = sprint(showerror, err)
        tag = on_error == unsupported ? "exception (provisionally unsupported; may be harness or upstream): " : "exception: "
        record!(rep, section, name, on_error, string(detail, tag, first(msg, 600)))
        return nothing
    end
end

"""
    validate_column_mlp(fixture_path; outdir = nothing, backends = DERIVATIVE_BACKENDS) -> Report

Run the complete validation of the native Lux column MLP against one fixture.
"""
function validate_column_mlp(fixture_path::AbstractString; outdir = nothing,
                             backends = DERIVATIVE_BACKENDS, rng = Xoshiro(0))
    fx = read_fixture(fixture_path)
    T = Torchlight.element_type(fx)
    rep = Report("Torchlight column MLP validation: $(basename(fixture_path))"; environment = _envinfo(fx))
    ftol, dtol, utol = forward_tolerance(T), derivative_tolerance(T), update_tolerance(T)
    n_layers = Int(fx.meta["n_layers"])

    # ---- mapping -----------------------------------------------------------
    t0 = time()
    model, ps, st, mapping, mrep = column_mlp_from_fixture(fx; rng)
    rep.timings["setup_and_mapping"] = time() - t0
    record!(rep, Evidence("mapping", "coverage", Torchlight.iscomplete(mrep) ? passed : failed, string(mrep),
                          Dict{String,Any}("n_source_keys" => mrep.n_source_keys, "n_leaves" => mrep.n_destination_leaves)))
    adapter = LuxAdapter(model, ps, st; name = "lux_native")

    # ---- forward -----------------------------------------------------------
    x, target = fx.input, fx.target
    t0 = time(); y, _ = forward(adapter, x); rep.timings["forward_plain_first"] = time() - t0
    t0 = time(); y, _ = forward(adapter, x); rep.timings["forward_plain_warm"] = time() - t0
    record!(rep, "forward", "output", compare(y, fx.output; name = "output", tol = ftol))
    inter = layer_intermediates(ps, x, n_layers)
    expected_stages = sort(collect(keys(inter)))          # every preact and hidden act
    have_stages = sort(collect(keys(fx.intermediates)))
    stages_complete = expected_stages == have_stages
    inter_results = [compare(inter[k], fx.intermediates[k]; name = k, tol = ftol) for k in have_stages if haskey(inter, k)]
    first_fail = findfirst(r -> !r.passed, inter_results)
    record!(rep, Evidence("forward", "intermediates", (stages_complete && all(r -> r.passed, inter_results)) ? passed : failed,
                          !stages_complete ? "fixture stages $have_stages != expected $expected_stages" :
                          first_fail === nothing ? "all $(length(inter_results)) stages match" :
                          "first failing stage: $(inter_results[first_fail])",
                          Dict{String,Any}("max_abs" => isempty(inter_results) ? NaN : maximum(r.max_abs for r in inter_results),
                                           "normalized_l2" => isempty(inter_results) ? NaN : maximum(r.normalized_l2 for r in inter_results),
                                           "n_fail" => sum((r.n_fail for r in inter_results); init = 0),
                                           "n_total" => sum((r.n_total for r in inter_results); init = 0),
                                           "atol" => ftol.atol, "rtol" => ftol.rtol, "relative_to_max" => false,
                                           "stages" => [r.name for r in inter_results])))
    for r in inter_results
        record!(rep, "forward/stages", r.name, r)
    end
    sx = sensitivity_check(xx -> first(forward(adapter, xx)), x; rng)
    record!(rep, "forward", "sensitivity_input", sx.changed ? passed : failed, "max output change $(sx.max_change)")
    ps_pert = Lux.fmap(p -> p .+ T(1e-2) .* randn(rng, T, size(p)), ps)
    yp, _ = forward(adapter, x, ps_pert)
    dp = maximum(abs, yp .- y)
    record!(rep, "forward", "sensitivity_parameters", dp > 0 ? passed : failed, "max output change $dp")

    # ---- domain ------------------------------------------------------------
    dom_results = ComparisonResult[]
    dom_names = sort([k for k in keys(fx.domain) if k != "attrs"])
    missing_dom = [k for k in REQUIRED_DOMAIN_CASES if k ∉ dom_names]
    for name in dom_names
        c = fx.domain[name]
        yi, _ = forward(adapter, c["input"])
        r = compare(yi, c["output"]; name, tol = ftol)
        push!(dom_results, r)
        record!(rep, "domain/cases", name, r; detail = "batch $(size(c["input"], 2))")
    end
    dom_ok = isempty(missing_dom) && !isempty(dom_results) && all(r -> r.passed, dom_results)
    record!(rep, Evidence("domain", "all_cases", dom_ok ? passed : failed,
                          isempty(missing_dom) ? "$(count(r -> r.passed, dom_results))/$(length(dom_results)) input families match" :
                                                 "fixture lacks required domain cases $missing_dom",
                          Dict{String,Any}("max_abs" => isempty(dom_results) ? NaN : maximum(r.max_abs for r in dom_results),
                                           "normalized_l2" => isempty(dom_results) ? NaN : maximum(r.normalized_l2 for r in dom_results),
                                           "n_fail" => sum((r.n_fail for r in dom_results); init = 0),
                                           "n_total" => sum((r.n_total for r in dom_results); init = 0),
                                           "atol" => ftol.atol, "rtol" => ftol.rtol, "relative_to_max" => false)))
    # column independence within Lux: split batch and permuted batch reproduce the joint result
    nb = size(x, 2); h = nb ÷ 2
    y_split = hcat(first(forward(adapter, x[:, 1:h])), first(forward(adapter, x[:, h+1:end])))
    perm = randperm(rng, nb)
    y_perm = first(forward(adapter, x[:, perm]))
    r1 = compare(y_split, y; name = "split_batch", tol = ftol)
    r2 = compare(y_perm, y[:, perm]; name = "permuted_batch", tol = ftol)
    record!(rep, "domain", "column_independence", (r1.passed && r2.passed) ? passed : failed, string(r1, "; ", r2))

    # ---- derivatives per backend ---------------------------------------------
    lossfn = yy -> mse_mean(yy, target)
    v = fx.probes["output_cotangent"]
    vjpfn = yy -> sum(v .* yy)
    d = fx.derivatives
    grads_by_backend = Dict{Symbol,Any}()
    for b in DERIVATIVE_BACKENDS
        b in backends && continue
        for n in ("mse_loss", "grad_input", "grad_params", "vjp_input", "vjp_params")
            record!(rep, "derivatives/$b", n, not_tested, "backend $b omitted from this run")
        end
        b === :reactant && record!(rep, "derivatives/reactant", "compiled_forward", not_tested, "backend reactant omitted from this run")
    end
    for b in backends
        sec = "derivatives/$b"
        if !backend_available(b)
            for n in ("mse_loss", "grad_input", "grad_params", "vjp_input", "vjp_params")
                record!(rep, sec, n, unsupported, "backend $b not available")
            end
            continue
        end
        res = _try(rep, sec, "mse_loss"; detail = "gradient failed: ") do
            t0 = time()
            out = b === :reactant ? gradients(adapter, Val(:reactant), lossfn, x; return_timings = true) :
                                    gradients(adapter, b, lossfn, x)
            rep.timings["grad_$(b)_first_total"] = time() - t0
            if b === :reactant
                rep.timings["grad_reactant_compile"] = out[4].compile
                rep.timings["grad_reactant_run_sync_first"] = out[4].run
                rep.timings["grad_reactant_run_sync_warm"] = out[4].warm
                out = out[1:3]
            else
                t0 = time(); gradients(adapter, b, lossfn, x); rep.timings["grad_$(b)_warm"] = time() - t0
            end
            out
        end
        res === nothing && (for n in ("grad_input", "grad_params", "vjp_input", "vjp_params")
                                record!(rep, sec, n, not_tested, "skipped: loss gradient failed")
                            end; continue)
        lval, gx, gps = res
        grads_by_backend[b] = (gx, gps)
        record!(rep, sec, "mse_loss", compare([T(lval)], [d["loss"]]; name = "mse_loss", tol = ftol))
        record!(rep, sec, "grad_input", compare(gx, d["grad_input"]; name = "grad_input", tol = dtol))
        _compare_params(rep, sec, "grad_params", gps, d["grad_params"], mapping, dtol)
        vres = _try(rep, sec, "vjp_input"; detail = "vjp failed: ") do
            b === :reactant ? gradients(adapter, Val(:reactant), vjpfn, x)[1:3] : gradients(adapter, b, vjpfn, x)
        end
        vres === nothing && (record!(rep, sec, "vjp_params", not_tested, "skipped: vjp failed"); continue)
        _, vx, vps = vres
        record!(rep, sec, "vjp_input", compare(vx, d["vjp_input"]; name = "vjp_input", tol = dtol))
        _compare_params(rep, sec, "vjp_params", vps, d["vjp_params"], mapping, dtol)
        if b === :reactant
            cres = _try(rep, sec, "compiled_forward"; detail = "compile failed: ") do
                f, xr, psr, str, tc = compiled_forward(adapter, x)
                rep.timings["forward_reactant_compile"] = tc
                t0 = time(); yr = Array(first(f(xr, psr, str))); rep.timings["forward_reactant_run_sync_first"] = time() - t0
                t0 = time(); Array(first(f(xr, psr, str))); rep.timings["forward_reactant_run_sync_warm"] = time() - t0
                # anti-constant check on the compiled function: perturbed inputs must change outputs
                x2 = x .+ T(1e-2) .* randn(rng, T, size(x))
                yr2 = Array(first(f(Reactant.to_rarray(x2), psr, str)))
                ps2 = Lux.fmap(p -> p .+ T(1e-2) .* randn(rng, T, size(p)), ps)
                yr3 = Array(first(f(xr, Reactant.to_rarray(ps2), str)))
                (yr, maximum(abs, yr2 .- yr), maximum(abs, yr3 .- yr))
            end
            if cres !== nothing
                yr, ch, chp = cres
                r = compare(yr, fx.output; name = "compiled_forward", tol = ftol)
                record!(rep, Evidence(sec, "compiled_forward", (r.passed && ch > 0 && chp > 0) ? passed : failed,
                        string(r, "; perturbed-input change ", ch, "; perturbed-parameter change ", chp), Dict{String,Any}("max_abs" => r.max_abs,
                        "normalized_l2" => r.normalized_l2, "n_fail" => r.n_fail, "n_total" => r.n_total,
                        "atol" => ftol.atol, "rtol" => ftol.rtol, "relative_to_max" => false)))
            end
        end
    end
    # cross-backend agreement (Julia-internal)
    if haskey(grads_by_backend, :zygote) && haskey(grads_by_backend, :reactant)
        r = compare(grads_by_backend[:reactant][1], grads_by_backend[:zygote][1]; name = "reactant_vs_zygote_grad_input", tol = dtol)
        record!(rep, "derivatives", "reactant_vs_zygote_grad_input", r)
    end

    # ---- probes: finite differences, JVP, adjoint consistency ------------------
    if haskey(grads_by_backend, :zygote)
        gx, gps = grads_by_backend[:zygote]
        u = fx.probes["input_direction"]
        f_in = xx -> mse_mean(first(forward(adapter, xx)), target)
        fd = finite_difference_check(f_in, x, u, dot(gx, u))
        fd_tol = T === Float64 ? 1e-6 : 1e-2
        record!(rep, Evidence("probes", "finite_difference_input", fd.best_relative < fd_tol ? passed : failed,
                "best h=$(fd.best_h) fd=$(fd.best_fd) analytic=$(fd.analytic) rel=$(fd.best_relative) (bar $fd_tol)",
                Dict{String,Any}("max_abs" => fd.best_error, "sweep" => fd.errors, "relative" => fd.best_relative)))
        # parameter direction: flatten the mapped direction tree
        pdirs = fx.probes["param_directions"]
        dir_ps = Lux.fmap(zero, ps)
        dir_ps, _ = map_parameters(dir_ps, pdirs, mapping; strict = true)
        θ = Lux.fmap(copy, ps)
        f_p = pp -> mse_mean(first(forward(adapter, x, pp)), target)
        analytic_p = sum(dot(l[2], r[2]) for (l, r) in zip(Torchlight._leaves(gps), Torchlight._leaves(dir_ps)))
        steps = T === Float64 ? (1e-3, 1e-4, 1e-5, 1e-6, 1e-7) : (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
        errs = Tuple{Float64,Float64,Float64}[]
        for hh in steps
            fp = f_p(Lux.fmap((a, b) -> a .+ T(hh) .* b, θ, dir_ps))
            fm = f_p(Lux.fmap((a, b) -> a .- T(hh) .* b, θ, dir_ps))
            fdv = Float64((fp - fm) / (2T(hh)))
            push!(errs, (hh, fdv, abs(fdv - Float64(analytic_p))))
        end
        i = argmin(e[3] for e in errs)
        relp = errs[i][3] / max(abs(Float64(analytic_p)), floatmin(Float64))
        record!(rep, Evidence("probes", "finite_difference_params", relp < fd_tol ? passed : failed,
                "best h=$(errs[i][1]) fd=$(errs[i][2]) analytic=$analytic_p rel=$relp (bar $fd_tol)",
                Dict{String,Any}("max_abs" => errs[i][3], "sweep" => errs, "relative" => relp)))
        # Cross-framework adjoint consistency: source JVP (J u) against Lux VJP (Jᵀ v)
        if haskey(d, "jvp_output_from_input")
            _, vx, vps = gradients(adapter, :zygote, vjpfn, x)
            ac = adjoint_consistency(v, d["jvp_output_from_input"], u, vx)
            acp_lhs = Float64(dot(v, d["jvp_output_from_params"]))
            acp_rhs = sum(dot(l[2], r[2]) for (l, r) in zip(Torchlight._leaves(vps), Torchlight._leaves(dir_ps)))
            acp_rel = abs(acp_lhs - acp_rhs) / max(abs(acp_lhs), abs(acp_rhs), floatmin(Float64))
            ac_tol = T === Float64 ? 1e-8 : 1e-4
            ok = ac.rel_error < ac_tol && acp_rel < ac_tol
            record!(rep, Evidence("probes", "adjoint_consistency_cross", ok ? passed : failed,
                    "input: <v,Ju>=$(ac.lhs) <Jᵀv,u>=$(ac.rhs) rel=$(ac.rel_error); params: $acp_lhs vs $acp_rhs rel=$acp_rel (bar $ac_tol)",
                    Dict{String,Any}("rel_input" => ac.rel_error, "rel_params" => acp_rel)))
        else
            record!(rep, "probes", "adjoint_consistency_cross", not_tested, "fixture has no JVP data")
        end
        # Lux-side JVP via Enzyme forward mode (diagnostic; not required)
        jres = _try(rep, "probes", "jvp_input_enzyme_forward"; detail = "Enzyme forward mode (optional diagnostic): ", on_error = unsupported) do
            fwd = xx -> first(forward(adapter, xx))
            Enzyme.autodiff(Enzyme.set_runtime_activity(Enzyme.Forward), Enzyme.Const(fwd), Enzyme.Duplicated(x, u))[1]
        end
        jres === nothing || record!(rep, "probes", "jvp_input_enzyme_forward",
                                    compare(jres, d["jvp_output_from_input"]; name = "jvp_input", tol = dtol))
    else
        for n in ("finite_difference_input", "finite_difference_params", "adjoint_consistency_cross")
            record!(rep, "probes", n, not_tested, "zygote gradients unavailable")
        end
    end

    # ---- training: one SGD step and one Adam step ------------------------------
    if haskey(fx.training, "sgd") && haskey(grads_by_backend, :zygote)
        tr = fx.training["sgd"]
        lr = T(tr["attrs"]["lr"])
        gx, gps = grads_by_backend[:zygote]
        _compare_params(rep, "training", "sgd_grads", gps, tr["grads"], mapping, dtol)
        opt = Optimisers.setup(Optimisers.Descent(lr), ps)
        _, ps_sgd = Optimisers.update(opt, ps, gps)
        _compare_params(rep, "training", "sgd_params_after", ps_sgd, tr["params_after"], mapping, utol;
                        detail = "lr=$lr; ")
        # Compare the *update* too (more sensitive than post-update parameters).
        Δ_lux = Lux.fmap((a, b) -> a .- b, ps_sgd, ps)
        Δ_ref = Dict(k => tr["params_after"][k] .- tr["params_before"][k] for k in keys(tr["params_after"]))
        _compare_params(rep, "training", "sgd_update_delta", Δ_lux, Δ_ref, mapping, dtol)
    else
        record!(rep, "training", "sgd_params_after", not_tested, "no sgd group or no gradients")
    end
    if haskey(fx.training, "adam") && haskey(grads_by_backend, :zygote)
        tr = fx.training["adam"]
        a = tr["attrs"]
        lr, b1, b2, ϵ = T(a["lr"]), T(a["beta1"]), T(a["beta2"]), T(a["eps"])
        a["weight_decay"] == 0 || record!(rep, "training", "adam_weight_decay", unsupported, "nonzero weight decay not mapped")
        gx, gps = grads_by_backend[:zygote]
        opt = Optimisers.setup(Optimisers.Adam(lr, (b1, b2), ϵ), ps)
        opt2, ps_adam = Optimisers.update(opt, ps, gps)
        _compare_params(rep, "training", "adam_params_after", ps_adam, tr["params_after"], mapping, utol;
                        detail = "lr=$lr betas=($b1,$b2) eps=$ϵ; ")
        Δ_lux = Lux.fmap((aa, bb) -> aa .- bb, ps_adam, ps)
        Δ_ref = Dict(k => tr["params_after"][k] .- tr["params_before"][k] for k in keys(tr["params_after"]))
        adam_ok = _compare_params(rep, "training", "adam_update_delta", Δ_lux, Δ_ref, mapping, dtol)
        if !adam_ok
            # Diagnostic: the first Adam step is lr * g / (|g| + ε).  Where |g| is
            # within a few orders of ε, Float32 rounding of g changes the update
            # at O(1) relative size in *both* frameworks.  Report how many failing
            # components have tiny reference gradients so the failure can be
            # attributed (ill-conditioning vs. a semantic optimizer mismatch).
            n_fail_total = 0; n_tiny = 0
            for (key, path, transform) in mapping
                a = Float64.(Torchlight._getleaf(Δ_lux, path)); e = Float64.(transform(Δ_ref[key]))
                g = abs.(Float64.(transform(tr["grads"][key])))
                denom = dtol.relative_to_max ? max.(abs.(e), maximum(abs, e)) : abs.(e)
                fails = .!(abs.(a .- e) .<= dtol.atol .+ dtol.rtol .* denom)
                n_fail_total += count(fails); n_tiny += count(fails .& (g .< 1e3 * Float64(ϵ)))
            end
            record!(rep, "training", "adam_update_delta_diagnosis", n_tiny == n_fail_total ? passed : failed,
                    "$n_tiny of $n_fail_total failing update components have |grad| < 1e3*eps=$(1e3*ϵ): " *
                    (n_tiny == n_fail_total ? "consistent with Float32 ill-conditioning of g/(|g|+eps) amplifying gradient roundoff; " *
                                              "rule equivalence must be shown separately by feeding the exported gradients to the optimizer" :
                                              "some failures are NOT explained by tiny gradients"))
        end
        # Optimizer state: Optimisers.Adam leaf state is (mt, vt, βt); torch stores exp_avg, exp_avg_sq, step.
        mt = Lux.fmap(l -> l.state[1], opt2; exclude = x -> x isa Optimisers.Leaf)
        vt = Lux.fmap(l -> l.state[2], opt2; exclude = x -> x isa Optimisers.Leaf)
        ref_m = Dict(k => tr["opt_state"][k]["exp_avg"] for k in keys(tr["opt_state"]))
        ref_v = Dict(k => tr["opt_state"][k]["exp_avg_sq"] for k in keys(tr["opt_state"]))
        okm = _compare_params(rep, "training", "adam_state_exp_avg", mt, ref_m, mapping, dtol)
        okv = _compare_params(rep, "training", "adam_state_exp_avg_sq", vt, ref_v, mapping, dtol)
        steps_ok = all(tr["opt_state"][k]["attrs"]["step"] == 1 for k in keys(tr["opt_state"]))
        record!(rep, "training", "adam_state", (okm && okv && steps_ok) ? passed : failed,
                "exp_avg, exp_avg_sq compared in Lux layout; source step counters all == 1: $steps_ok")
        # Compiled training step through Lux.Training (Reactant + Enzyme): parameters after one SGD step.
        if :reactant in backends && haskey(fx.training, "sgd")
            tres = _try(rep, "training", "compiled_sgd_step_reactant"; detail = "compiled train step (optional diagnostic): ", on_error = unsupported) do
                lr_sgd = T(fx.training["sgd"]["attrs"]["lr"])
                xr, tr_ = Reactant.to_rarray(x), Reactant.to_rarray(target)
                ts = Lux.Training.TrainState(model, Reactant.to_rarray(ps), Reactant.to_rarray(st), Optimisers.Descent(lr_sgd))
                lossf = (m, p, s, (xx, tt)) -> (mse_mean(first(m(xx, p, s)), tt), s, (;))
                t0 = time()
                _, lval, _, ts2 = Lux.Training.single_train_step!(Lux.AutoEnzyme(), lossf, (xr, tr_), ts)
                rep.timings["train_step_reactant_first_total"] = time() - t0
                Torchlight._to_host(ts2.parameters)
            end
            tres === nothing || _compare_params(rep, "training", "compiled_sgd_step_reactant", tres,
                                                fx.training["sgd"]["params_after"], mapping, utol)
        end
    else
        record!(rep, "training", "adam_params_after", not_tested, "no adam group or no gradients")
        record!(rep, "training", "adam_state", not_tested, "no adam group or no gradients")
    end

    # ---- dropout: train-mode fixture with shared masks ----------------------------
    train_path = replace(fixture_path, r"\.h5$" => "_train.h5")
    if isfile(train_path)
        fxt = read_fixture(train_path)
        model_t, ps_t, st_t, mapping_t, _ = column_mlp_from_fixture(fxt; masked = true, rng)
        yt, _ = model_t(fxt.input, ps_t, st_t)
        record!(rep, "dropout", "masked_forward", compare(yt, fxt.output; name = "masked_forward", tol = ftol);
                detail = "shared 0/1 masks, scale 1/(1-p)")
        ad_t = LuxAdapter(model_t, ps_t, st_t)
        lt, gxt, gpst = gradients(ad_t, :zygote, yy -> mse_mean(yy, fxt.target), fxt.input)
        record!(rep, "dropout", "masked_grad_input", compare(gxt, fxt.derivatives["grad_input"]; name = "masked_grad_input", tol = dtol))
        _compare_params(rep, "dropout", "masked_grad_params", gpst, fxt.derivatives["grad_params"], mapping_t, dtol)
        # native Lux Dropout statistics (in-framework only): keep fraction and scaling
        p = Float64(fx.meta["dropout_p"])
        dl = Dropout(p); _, sd = Lux.setup(Xoshiro(1), dl)
        z = ones(T, 512, 256)
        zd, _ = dl(z, NamedTuple(), sd)
        keep = count(!iszero, zd) / length(zd)
        scale_ok = all(v -> v == 0 || isapprox(v, 1 / (1 - p); rtol = 1e-6), zd)
        record!(rep, "dropout", "native_dropout_statistics", (abs(keep - (1 - p)) < 0.01 && scale_ok) ? passed : failed,
                "keep fraction $keep (expected $(1-p) ± 0.01), scaling 1/(1-p): $scale_ok")
        zt, _ = dl(z, NamedTuple(), Lux.testmode(sd))
        record!(rep, "dropout", "native_dropout_testmode_identity", zt == z ? passed : failed, "test mode is identity")
        record!(rep, "dropout", "native_stochastic_trajectory_match", unsupported,
                "framework RNG streams differ by design; compare stochastic training statistically (not run here)")
    else
        record!(rep, "dropout", "masked_forward", not_tested, "no train-mode fixture at $train_path")
    end

    # ---- defect detection: the harness must flag injected mistakes -----------------
    perm_feat = randperm(rng, size(x, 1))
    r = compare(first(forward(adapter, x[perm_feat, :])), fx.output; name = "permuted_features", tol = ftol)
    record!(rep, "defect_detection", "permuted_features", r.passed ? failed : passed, "injected feature permutation flagged: $(!r.passed); $r")
    ps_nobias = merge(ps, (; dense_0 = merge(ps.dense_0, (; bias = zero(ps.dense_0.bias)))))
    r = compare(first(forward(adapter, x, ps_nobias)), fx.output; name = "missing_bias", tol = ftol)
    record!(rep, "defect_detection", "missing_bias", r.passed ? failed : passed, "zeroed first bias flagged: $(!r.passed); $r")
    wrong_loss = sum(abs2, y .- target) / size(y, 2)     # sum over features, mean over batch
    r = compare([T(wrong_loss)], [d["loss"]]; name = "wrong_loss_denominator", tol = ftol)
    record!(rep, "defect_detection", "wrong_loss_denominator", r.passed ? failed : passed, "wrong reduction flagged: $(!r.passed); $r")
    r = compare(zero(d["grad_input"]), d["grad_input"]; name = "detached_input_gradient", tol = dtol)
    record!(rep, "defect_detection", "detached_input_gradient", r.passed ? failed : passed, "zero input gradient flagged: $(!r.passed); $r")
    # mapping defects: a missing key and a swapped destination must raise
    bad = copy(mapping); pop!(bad)
    threw = try; map_parameters(Lux.f64(ps), fx.state_dict, bad; strict = true); false; catch; true; end
    record!(rep, "defect_detection", "incomplete_mapping_raises", threw ? passed : failed, "dropping one mapping entry raises: $threw")

    ok, missing, not_passed = acceptance(rep, REQUIRED)
    record!(rep, "acceptance", "required_capabilities", ok ? passed : failed,
            ok ? "all $(length(REQUIRED)) required capabilities passed and no failures" :
                 "missing: $missing; not passed: $not_passed; failed evidence: $(summarize(rep)[failed])")
    if outdir !== nothing
        mkpath(outdir)
        stem = replace(basename(fixture_path), r"\.h5$" => "")
        write_markdown(joinpath(outdir, "$stem.md"), rep)
        write_json(joinpath(outdir, "$stem.json"), rep)
    end
    return rep
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    using .ColumnMLPValidate, Torchlight
    path = ARGS[1]
    outdir = length(ARGS) ≥ 2 ? ARGS[2] : joinpath(@__DIR__, "..", "..", "benchmarks", "results")
    rep = validate_column_mlp(path; outdir)
    c = summarize(rep)
    println("passed=$(c[passed]) failed=$(c[failed]) unsupported=$(c[unsupported]) not_tested=$(c[not_tested])")
    ok, _, _ = acceptance(rep, ColumnMLPValidate.REQUIRED)
    println(ok ? "ACCEPTED" : "NOT ACCEPTED")
    for e in rep.evidence
        e.status in (failed, unsupported) && println("  ", e.status, " ", e.section, "/", e.name, ": ", first(e.detail, 300))
    end
    exit(ok ? 0 : 1)
end
