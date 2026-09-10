# Independent failure-injection checks from implementation review.
using HDF5
using Lux, Random

module ReviewColumnMLP
include(joinpath(@__DIR__, "..", "cases", "column_mlp", "ColumnMLP.jl"))
end

@testset "Review: fixed-mask dropout contract" begin
    case = ReviewColumnMLP.ColumnMLPCase
    for probability in (-0.1, 1.0, NaN)
        @test_throws Exception case.column_mlp([3, 5, 2]; dropout_p = probability)
        @test_throws Exception case.FixedMaskDropout(probability)
    end
    model = case.column_mlp([3, 5, 2]; dropout_p = 0.1, masked = true)
    ps, st = Lux.setup(Xoshiro(48), model)
    x = ones(Float32, 3, 7)
    installed = case._install_masks(st, Dict("layer0" => ones(Float32, 5, 7)), 1)
    y, _ = model(x, ps, installed)
    @test size(y) == (2, 7)
    @test all(isfinite, y)
    @test_throws Exception case._install_masks(st, Dict("layer0" => fill(0.5f0, 5, 7)), 1)
    @test_throws Exception case._install_masks(st, Dict("layer0" => ones(Float32, 5, 7), "layer1" => ones(Float32, 5, 7)), 1)
    @test_throws Exception case._install_masks(st, Dict{String,Any}(), 1)
    bad_shape = case._install_masks(st, Dict("layer0" => ones(Float32, 5, 1)), 1)
    @test_throws Exception model(x, ps, bad_shape)
    @test_throws Exception case.mse_mean(ones(2, 7), ones(2, 1))
end

@testset "Review: compare cannot certify invalid evidence" begin
    @test Torchlight.compare([1.0], [1.0]).passed
    @test Torchlight.compare(1.0, 1.0).passed
    @test !Torchlight.compare(zeros(0, 0), zeros(0, 0)).passed
    @test !Torchlight.compare(Float32[1;;], Float64[1;;]).passed
    @test !Torchlight.compare(ones(1, 2), ones(2, 1)).passed
    # A large unrelated component must not conceal an error near zero.
    @test !Torchlight.compare([1e12 1.0], [1e12 0.0]).passed
    @test !Torchlight.compare([Inf;;], [Inf;;]).passed
    @test !Torchlight.compare([NaN;;], [NaN;;]).passed
    @test !Torchlight.compare([0.0;;], [NaN;;]).passed
end

@testset "Review: insert exactly the parameter array that was validated" begin
    calls = Ref(0)
    function transform_once(array)
        calls[] += 1
        return calls[] == 1 ? copy(array) : zeros(Float32, 1, 1)
    end
    template = (; weight = zeros(2, 3))
    source = Dict("w" => ones(2, 3))
    mapped, report = Torchlight.map_parameters(template, source,
        [("w", (:weight,), transform_once)])
    @test calls[] == 1
    @test Torchlight.iscomplete(report)
    @test size(mapped.weight) == (2, 3)
    @test eltype(mapped.weight) === Float64
    @test mapped.weight == ones(2, 3)
    @test template.weight == zeros(2, 3)
    @test_throws Exception Torchlight.map_parameters(template, source,
        [("w", (:weight,), identity), ("w", (:weight,), identity)])
    @test_throws Exception Torchlight.map_parameters(template, source, [])
end

function review_fixture(path; defect = :none)
    h5open(path, "w") do file
        meta = create_group(file, "meta")
        for (key, value) in ("schema_version" => 1, "dtype" => "float64",
                            "source_framework" => "torch", "source_version" => "review",
                            "model_id" => "review_dense", "mode" => "eval",
                            "activation" => "tanh", "output_activation" => "identity",
                            "loss" => "mse_mean", "n_layers" => 1, "widths" => [3, 2])
            attrs(meta)[key] = value
        end
        file["input"] = ones(3, 2)
        file["output"] = ones(2, 2)
        file["target"] = defect === :target_dtype ? ones(Float32, 2, 2) : ones(2, 2)
        state = create_group(file, "state_dict")
        state["layers.0.weight"] = defect === :nonfinite_weight ? fill(NaN, 3, 2) : ones(3, 2)
        state["layers.0.bias"] = ones(2)
        defect === :integer_buffer && (state["counter"] = Int64[7])
        defect === :scalar_integer_buffer && (state["counter"] = Int64(7))
        if defect === :intermediate_dtype
            create_group(file, "intermediates")["layer0_preact"] = ones(Float32, 2, 2)
        elseif defect === :probe_dtype
            create_group(file, "probes")["output_cotangent"] = ones(Float32, 2, 2)
        elseif defect === :nonfinite_derivative
            create_group(file, "derivatives")["grad_input"] = fill(Inf, 3, 2)
        end
    end
end

@testset "Review: fixture precision and finite-value contract" begin
    mktempdir() do dir
        path = joinpath(dir, "reference.h5")
        review_fixture(path)
        @test eltype(Torchlight.read_fixture(path).input) === Float64
        review_fixture(path; defect = :integer_buffer)
        fixture = Torchlight.read_fixture(path)
        @test fixture.state_dict["counter"] == Int64[7]
        @test eltype(fixture.state_dict["counter"]) === Int64
        review_fixture(path; defect = :scalar_integer_buffer)
        fixture = Torchlight.read_fixture(path)
        @test size(fixture.state_dict["counter"]) == ()
        @test fixture.state_dict["counter"][] === Int64(7)
        for defect in (:target_dtype, :nonfinite_weight, :intermediate_dtype,
                       :probe_dtype, :nonfinite_derivative)
            review_fixture(path; defect)
            @test_throws Exception Torchlight.read_fixture(path)
        end
    end
end

@testset "Review: required coverage controls acceptance" begin
    required = [("forward", "reference"), ("derivatives", "input")]
    report = Torchlight.Report("Required coverage")
    Torchlight.record!(report, "forward", "reference", Torchlight.passed)
    ok, missing, not_passed = Torchlight.acceptance(report, required)
    @test !ok
    @test missing == ["derivatives / input"]
    for status in (Torchlight.not_tested, Torchlight.unsupported, Torchlight.failed)
        r = deepcopy(report)
        Torchlight.record!(r, "derivatives", "input", status)
        @test !first(Torchlight.acceptance(r, required))
    end
    Torchlight.record!(report, "derivatives", "input", Torchlight.passed)
    @test first(Torchlight.acceptance(report, required))
    # A duplicate success must not overwrite a prior failure or omission.
    Torchlight.record!(report, "derivatives", "input", Torchlight.not_tested)
    @test !first(Torchlight.acceptance(report, required))
end

@testset "Review: reports disclose tolerance policy and escape errors" begin
    result = Torchlight.compare([1.0;;], [1.0;;])
    evidence = Torchlight.Evidence("forward", "sentinel", result)
    @test haskey(evidence.metrics, "relative_to_max")
    @test evidence.metrics["relative_to_max"] === false
    report = Torchlight.Report("Control characters")
    Torchlight.record!(report, "failure", "exception", Torchlight.failed, "line1\r\nline2\b\f\0")
    io = IOBuffer()
    Torchlight.write_json(io, report)
    serialized = String(take!(io))
    # Raw characters below U+0020 are illegal within JSON strings.
    @test !occursin('\r', serialized)
    @test !occursin('\b', serialized)
    @test !occursin('\f', serialized)
    @test !occursin('\0', serialized)
    @test occursin("\\r", serialized) || occursin("\\u000d", serialized)
end
