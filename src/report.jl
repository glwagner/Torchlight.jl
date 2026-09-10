# Evidence records and reports.

"""
    CapabilityStatus

`passed`, `failed`, `unsupported` (the route/backend cannot express the
capability; recorded with the exact limitation), or `not_tested` (not yet
executed; never inferred).
"""
@enum CapabilityStatus passed failed unsupported not_tested

struct Evidence
    section::String
    name::String
    status::CapabilityStatus
    detail::String
    metrics::Dict{String,Any}
end

Evidence(section, name, status, detail = "") = Evidence(section, name, status, detail, Dict{String,Any}())

function Evidence(section, name, r::ComparisonResult; detail = "")
    m = Dict{String,Any}("max_abs" => r.max_abs, "max_rel" => r.max_rel, "normalized_l2" => r.normalized_l2,
                         "n_fail" => r.n_fail, "n_total" => r.n_total, "ref_scale" => r.ref_scale,
                         "atol" => r.tol.atol, "rtol" => r.tol.rtol, "relative_to_max" => r.tol.relative_to_max,
                         "nonfinite" => r.nonfinite, "dtype_mismatch" => r.dtype_mismatch,
                         "shape_actual" => collect(r.shape_actual), "shape_expected" => collect(r.shape_expected),
                         "worst_index" => r.worst_index === nothing ? nothing : collect(r.worst_index),
                         "worst_actual" => r.worst_actual, "worst_expected" => r.worst_expected)
    return Evidence(section, name, r.passed ? passed : failed, detail, m)
end

"""
    Report(title; environment = Dict())

Ordered collection of `Evidence`.  `record!(report, evidence)` appends;
`summarize(report)` returns counts by status; `write_markdown`/`write_json`
render it.  Reports never drop evidence: a failing check stays in the report.
"""
mutable struct Report
    title::String
    created::String
    environment::Dict{String,Any}
    evidence::Vector{Evidence}
    timings::Dict{String,Float64}
end

Report(title; environment = Dict{String,Any}()) =
    Report(String(title), string(Libc.strftime("%Y-%m-%dT%H:%M:%S", time())), Dict{String,Any}(environment),
           Evidence[], Dict{String,Float64}())

record!(r::Report, e::Evidence) = (push!(r.evidence, e); e)
record!(r::Report, section, name, status::CapabilityStatus, detail = "") =
    record!(r, Evidence(section, name, status, detail))
record!(r::Report, section, name, c::ComparisonResult; detail = "") =
    record!(r, Evidence(section, name, c; detail))

function summarize(r::Report)
    counts = Dict(s => 0 for s in instances(CapabilityStatus))
    for e in r.evidence
        counts[e.status] += 1
    end
    return counts
end

"""`true` when no evidence is `failed`.  Says nothing about `not_tested` or
`unsupported`; use `acceptance` for that."""
no_failures(r::Report) = summarize(r)[failed] == 0

"""
    acceptance(r::Report, required) -> (ok::Bool, missing::Vector{String}, not_passed::Vector{String})

Acceptance requires no `failed` evidence **and** every `(section, name)` pair in
`required` to be present with status `passed`.  A required capability that was
never recorded, or recorded as `unsupported`/`not_tested`, fails acceptance.
"""
function acceptance(r::Report, required)
    missing = String[]
    not_passed = String[]
    for (sec, name) in required
        idx = findall(e -> e.section == sec && e.name == name, r.evidence)
        if isempty(idx)
            push!(missing, "$sec / $name")
        elseif any(r.evidence[i].status != passed for i in idx)
            push!(not_passed, "$sec / $name")
        end
    end
    ok = no_failures(r) && isempty(missing) && isempty(not_passed)
    return ok, missing, not_passed
end

_fmt(x::Real) = isinteger(x) && abs(x) < 1e15 ? string(Int(x)) : @sprintf("%.3e", x)
_fmt(x) = string(x)

function write_markdown(io::IO, r::Report)
    c = summarize(r)
    println(io, "# ", r.title, "\n")
    println(io, "Generated ", r.created, ".\n")
    println(io, "| passed | failed | unsupported | not_tested |")
    println(io, "|---|---|---|---|")
    println(io, "| ", c[passed], " | ", c[failed], " | ", c[unsupported], " | ", c[not_tested], " |\n")
    if !isempty(r.environment)
        println(io, "## Environment\n")
        for k in sort(collect(keys(r.environment)))
            println(io, "- `", k, "`: ", r.environment[k])
        end
        println(io)
    end
    if !isempty(r.timings)
        println(io, "## Timings (seconds)\n")
        println(io, "| step | seconds |\n|---|---|")
        for k in sort(collect(keys(r.timings)))
            @printf(io, "| %s | %.3f |\n", k, r.timings[k])
        end
        println(io)
    end
    sections = unique(e.section for e in r.evidence)
    for s in sections
        println(io, "## ", s, "\n")
        println(io, "| check | status | max_abs | normalized_l2 | fails | tol (atol, rtol) | detail |")
        println(io, "|---|---|---|---|---|---|---|")
        for e in r.evidence
            e.section == s || continue
            m = e.metrics
            ma = haskey(m, "max_abs") ? _fmt(m["max_abs"]) : ""
            nl = haskey(m, "normalized_l2") ? _fmt(m["normalized_l2"]) : ""
            nf = haskey(m, "n_fail") ? string(m["n_fail"], "/", m["n_total"]) : ""
            tl = haskey(m, "atol") ? string("(", _fmt(m["atol"]), ", ", _fmt(m["rtol"]), ")") : ""
            println(io, "| ", e.name, " | **", e.status, "** | ", ma, " | ", nl, " | ", nf, " | ", tl, " | ",
                    replace(e.detail, "\n" => " ", "|" => "\\|"), " |")
        end
        println(io)
    end
end
write_markdown(path::AbstractString, r::Report) = open(io -> write_markdown(io, r), path, "w")

# Minimal JSON writer (no external dependency) for report serialization.
_json(io::IO, x::Nothing) = print(io, "null")
_json(io::IO, x::Bool) = print(io, x ? "true" : "false")
_json(io::IO, x::Integer) = print(io, x)
_json(io::IO, x::AbstractFloat) = isfinite(x) ? print(io, repr(Float64(x))) : print(io, "\"", x, "\"")
_json(io::IO, x::CapabilityStatus) = _json(io, string(x))
_json(io::IO, x::Symbol) = _json(io, string(x))
function _json(io::IO, s::AbstractString)
    print(io, '"')
    for ch in s
        if ch == '"'
            print(io, "\\\"")
        elseif ch == '\\'
            print(io, "\\\\")
        elseif ch == '\n'
            print(io, "\\n")
        elseif ch == '\r'
            print(io, "\\r")
        elseif ch == '\t'
            print(io, "\\t")
        elseif ch == '\b'
            print(io, "\\b")
        elseif ch == '\f'
            print(io, "\\f")
        elseif UInt32(ch) < 0x20 || UInt32(ch) == 0x7f
            @printf(io, "\\u%04x", UInt32(ch))
        else
            print(io, ch)
        end
    end
    print(io, '"')
end
function _json(io::IO, v::Union{AbstractVector,Tuple})
    print(io, "[")
    for (i, x) in enumerate(v)
        i > 1 && print(io, ",")
        _json(io, x)
    end
    print(io, "]")
end
function _json(io::IO, d::AbstractDict)
    print(io, "{")
    for (i, k) in enumerate(sort(collect(keys(d)); by = string))
        i > 1 && print(io, ",")
        _json(io, string(k)); print(io, ":"); _json(io, d[k])
    end
    print(io, "}")
end
_json(io::IO, e::Evidence) = _json(io, Dict("section" => e.section, "name" => e.name, "status" => e.status,
                                            "detail" => e.detail, "metrics" => e.metrics))
_json(io::IO, x) = _json(io, string(x))

function write_json(io::IO, r::Report)
    _json(io, Dict("title" => r.title, "created" => r.created, "environment" => r.environment,
                   "summary" => Dict(string(k) => v for (k, v) in summarize(r)),
                   "timings" => r.timings, "evidence" => r.evidence))
    println(io)
end
write_json(path::AbstractString, r::Report) = open(io -> write_json(io, r), path, "w")
