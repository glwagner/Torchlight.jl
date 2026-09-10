using Test
using Torchlight
using Lux, Random, HDF5, LinearAlgebra

const FIXTURES = joinpath(@__DIR__, "fixtures")
const TINY64 = joinpath(FIXTURES, "column_mlp_tiny_float64.h5")
const TINY32 = joinpath(FIXTURES, "column_mlp_tiny_float32.h5")

@testset "Torchlight" begin
    @testset "compare" begin
        e = [1.0 2.0; 3.0 4.0]
        r = compare(e, e; name = "identity")
        @test r.passed && r.n_fail == 0 && r.max_abs == 0
        r = compare(e .+ 1e-6, e; atol = 1e-10, rtol = 1e-8)
        @test !r.passed && r.n_fail == 4
        @test compare(e .+ 1e-6, e; atol = 1e-5, rtol = 0).passed
        # shape, dtype, nonfinite, empty all fail
        @test !compare(ones(2, 3), ones(3, 2)).passed
        @test !compare(Float32.(e), e).passed
        @test compare(Float32.(e), e; allow_dtype_mismatch = true).passed
        @test !compare([NaN 2.0; 3.0 4.0], e).passed
        @test !compare(zeros(0), zeros(0)).passed
        # relative_to_max is off by default and recorded when on
        big = [1e3, 1e-9]
        act = [1e3, 2e-9]
        @test !compare(act, big; atol = 0, rtol = 1e-8).passed
        rr = compare(act, big; atol = 0, rtol = 1e-8, relative_to_max = true)
        @test rr.passed && rr.tol.relative_to_max
        @test_throws ErrorException compare(e, e; tol = Tolerance(1, 1), atol = 1)
        # vector and scalar comparisons
        @test compare([1.0, 2.0], [1.0, 2.0]).worst_index == (1,)
        @test compare(1.0, 1.0).passed
    end

    @testset "fixture reader" begin
        fx = read_fixture(TINY64)
        @test fx.meta["schema_version"] == 1
        @test eltype(fx.input) === Float64 && size(fx.input) == (7, 13)   # (features, batch)
        @test size(fx.state_dict["layers.0.weight"]) == (7, 11)            # HDF5-natural (in, out)
        @test size(to_lux_dense_weight(fx.state_dict["layers.0.weight"])) == (11, 7)
        @test eltype(reverse_axes(fx.state_dict["layers.0.weight"])) === Float64
        @test dense_weight_transform(fx) === to_lux_dense_weight
        @test Torchlight.element_type(fx) === Float64
        @test fx.training["adam"]["opt_state"]["layers.0.weight"]["attrs"]["step"] == 1
        fx32 = read_fixture(TINY32)
        @test eltype(fx32.output) === Float32
        # logical index check against a known element written by numpy: a[i,j] -> julia a[j+1,i+1]
        h5open(TINY64, "r") do f
            w = read(f["state_dict/layers.0.weight"])
            @test w[3, 2] == fx.state_dict["layers.0.weight"][3, 2]
        end
        # malformed fixtures are rejected
        tmp = tempname() * ".h5"
        h5open(tmp, "w") do f
            g = create_group(f, "meta"); attributes(g)["schema_version"] = 99
            write(f, "input", ones(2, 2)); write(f, "output", ones(2, 2)); create_group(f, "state_dict")
        end
        @test_throws ErrorException read_fixture(tmp)
        rm(tmp)
        @test_throws ErrorException read_fixture(tempname())
    end

    @testset "map_parameters" begin
        fx = read_fixture(TINY64)
        model = Chain(; dense_0 = Dense(7 => 11, tanh), dense_1 = Dense(11 => 5, tanh), dense_2 = Dense(5 => 3))
        ps, _ = Lux.setup(Xoshiro(0), model)
        ps = Lux.f64(ps)
        wt = dense_weight_transform(fx)
        mapping = Torchlight.ParameterMapping()
        for i in 0:2
            push!(mapping, ("layers.$i.weight", (Symbol("dense_$i"), :weight), wt))
            push!(mapping, ("layers.$i.bias", (Symbol("dense_$i"), :bias), identity))
        end
        ps2, rep = map_parameters(ps, fx.state_dict, mapping)
        @test Torchlight.iscomplete(rep)
        @test ps2.dense_0.weight == to_lux_dense_weight(fx.state_dict["layers.0.weight"])
        @test ps2.dense_2.bias == fx.state_dict["layers.2.bias"]
        # incomplete: drop an entry -> unused source key and unmapped destination
        @test_throws ErrorException map_parameters(ps, fx.state_dict, mapping[1:end-1])
        _, rep2 = map_parameters(ps, fx.state_dict, mapping[1:end-1]; strict = false)
        @test rep2.unused_source_keys == ["layers.2.bias"] && rep2.unmapped_destinations == ["dense_2.bias"]
        # duplicate destination
        dup = vcat(mapping, [("layers.0.bias", (:dense_0, :bias), identity)])
        _, rep3 = map_parameters(ps, fx.state_dict, dup; strict = false)
        @test rep3.duplicate_destinations == ["dense_0.bias"]
        # wrong transform -> shape mismatch; Float32 template -> dtype mismatch
        bad = copy(mapping); bad[1] = ("layers.0.weight", (:dense_0, :weight), identity)
        @test_throws ErrorException map_parameters(ps, fx.state_dict, bad)
        @test_throws ErrorException map_parameters(Lux.f32(ps), fx.state_dict, mapping)
        # unknown destination and missing source key
        @test_throws ErrorException map_parameters(ps, fx.state_dict, [("layers.0.weight", (:nope, :weight), wt)])
        @test_throws ErrorException map_parameters(ps, fx.state_dict, [("missing", (:dense_0, :weight), wt)])
        # transforms are evaluated exactly once
        calls = Ref(0)
        counting = w -> (calls[] += 1; wt(w))
        m2 = [(k, p, t === wt ? counting : t) for (k, p, t) in mapping]
        map_parameters(ps, fx.state_dict, m2)
        @test calls[] == 3
    end

    @testset "report" begin
        rep = Report("t")
        record!(rep, "s", "a", passed, "ok")
        record!(rep, "s", "b", compare([1.0], [2.0]))
        record!(rep, "s", "c", not_tested)
        c = summarize(rep)
        @test c[passed] == 1 && c[failed] == 1 && c[not_tested] == 1
        @test !no_failures(rep)
        ok, missing, np = acceptance(rep, [("s", "a"), ("s", "c"), ("s", "zzz")])
        @test !ok && missing == ["s / zzz"] && np == ["s / c"]
        io = IOBuffer(); write_markdown(io, rep); md = String(take!(io))
        @test occursin("**failed**", md)
        io = IOBuffer(); write_json(io, rep); js = String(take!(io))
        @test occursin("\"relative_to_max\":false", js)
        # control characters are escaped
        record!(rep, "s", "d", failed, "line\r\nbreak\x01")
        io = IOBuffer(); write_json(io, rep); js = String(take!(io))
        @test occursin("\\r\\n", js) && occursin("\\u0001", js)
    end

    @testset "probes" begin
        f = x -> sum(abs2, x)
        x = [1.0, 2.0]; v = [0.5, -1.0]
        fd = finite_difference_check(f, x, v, dot(2x, v))
        @test fd.best_relative < 1e-8
        ac = adjoint_consistency([1.0, 2.0], [3.0, 4.0], [5.0, 6.0, 7.0], [0.5, 0.5, 1.0])
        @test ac.lhs == 11.0 && ac.rhs == 12.5
        s = sensitivity_check(x -> zeros(2), x)
        @test !s.changed
    end

    @testset "column MLP acceptance (tiny Float64)" begin
        include(joinpath(@__DIR__, "..", "cases", "column_mlp", "validate.jl"))
        backends = Tuple(Symbol.(split(get(ENV, "TORCHLIGHT_TEST_BACKENDS", "zygote,enzyme,reactant"), ",")))
        outdir = mktempdir()
        rep = ColumnMLPValidate.validate_column_mlp(TINY64; outdir, backends)
        c = summarize(rep)
        @test c[failed] == 0
        ok, missing, np = acceptance(rep, ColumnMLPValidate.REQUIRED)
        if backends == (:zygote, :enzyme, :reactant)
            @test ok
        else
            @test !ok    # omitted backends are not_tested, never passed by default
            @test c[not_tested] > 0
        end
        @test isfile(joinpath(outdir, "column_mlp_tiny_float64.md"))
        # defect detection must have been exercised and passed
        for n in ("permuted_features", "missing_bias", "wrong_loss_denominator", "detached_input_gradient", "incomplete_mapping_raises")
            e = only(filter(e -> e.section == "defect_detection" && e.name == n, rep.evidence))
            @test e.status == passed
        end
        # Float32 tiny fixture: forward parity with strict tolerance
        rep32 = ColumnMLPValidate.validate_column_mlp(TINY32; backends = (:zygote,))
        @test only(filter(e -> e.section == "forward" && e.name == "output", rep32.evidence)).status == passed
    end

    include("review_regressions.jl")
end
