# Symmetric-root check for jparity: seed the Julia model at an EXTERNAL point
# (equilibria's solved solution, given as a CSV) with tms shocked, solve, and dump
# the result. If Julia stays at the seeded point, equilibria's root is one Julia
# admits; if Julia moves away, equilibria converged to a root Julia rejects (a
# closure/formulation discrepancy, not basin noise).
#
# Args: <dataset> <tariff_power> <seed_csv> <out_csv>
#   seed_csv: rows "name,idx...,value" holding a full variable point to seed from.

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

dataset      = ARGS[1]
tariff_power = parse(Float64, ARGS[2])
seed_csv     = ARGS[3]
out_csv      = ARGS[4]

(; hData, hParameters, hSets) = get_sample_data()
mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
sd = deepcopy(mc.data)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, sd)
fd = deepcopy(mc.fixed)
mc.data = data_calibration; mc.fixed = fixed_calibration
run_model!(mc)
rebuild_model!(mc)
mc.fixed = deepcopy(fd)

# --- overwrite mc.data with the external seed point (equilibria's solution) ---
seed = Dict{String, Dict{Tuple, Float64}}()
for line in eachline(seed_csv)
    (startswith(line, ">>>") || isempty(line)) && continue
    parts = split(line, ",")
    name = parts[1]
    val = tryparse(Float64, parts[end])
    val === nothing && continue
    idx = Tuple(parts[2:end-1])
    haskey(seed, name) || (seed[name] = Dict{Tuple, Float64}())
    seed[name][idx] = val
end
for (name, cells) in seed
    haskey(mc.data, name) || continue
    na = mc.data[name]
    isa(na, Number) && continue
    for idx in CartesianIndices(size(na))
        key = Tuple(string(names(na, d)[idx[d]]) for d in 1:ndims(na))
        haskey(cells, key) && (na[idx] = cells[key])
    end
end

# apply the shock and solve ONE pass from the seeded point
for c in hSets["comm"], s in hSets["reg"], d in hSets["reg"]
    try; mc.data["tms"][c, s, d] = tariff_power; catch; end
end
run_model!(mc)

function dk(io, k)
    v = mc.data[k]
    if isa(v, Number)
        println(io, "$k,$v")
    else
        for idx in CartesianIndices(size(v))
            key = join([string(names(v, d)[idx[d]]) for d in 1:ndims(v)], ",")
            println(io, "$k,$key,$(v[idx])")
        end
    end
end
open(out_csv, "w") do io
    for k in sort(collect(keys(mc.data))); try; dk(io, k); catch; end; end
end
println(">>> DONE")
