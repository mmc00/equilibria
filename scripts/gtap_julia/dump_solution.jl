# Dump Julia's full BASE SOLUTION (all mc.data vars after a base solve) plus all
# mc.parameters (elasticities), as the reference for the per-equation residual
# tests. Args: <dataset> <out_csv>. CSV rows: key,idx...,value (scalar: key,value).

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

dataset = length(ARGS) >= 1 ? ARGS[1] : "sample"
out_csv = length(ARGS) >= 2 ? ARGS[2] : "julia_solution.csv"

if dataset == "sample"
    (; hData, hParameters, hSets) = get_sample_data()
else
    import HeaderArrayFile
    data_har = HeaderArrayFile.readHar(joinpath(dataset, "gsdfdat.har"))
    sets_har = HeaderArrayFile.readHar(joinpath(dataset, "gsdfset.har"))
    par_har  = HeaderArrayFile.readHar(joinpath(dataset, "gsdfpar.har"))
    hData       = Dict(keys(data_har) .=> NamedArray.(values(data_har)))
    hSets       = Dict(keys(sets_har) .=> NamedArray.(values(sets_har)))
    hParameters = Dict(keys(par_har)  .=> NamedArray.(values(par_har)))
end

mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
start_data = deepcopy(mc.data)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, start_data)
fixed_default = deepcopy(mc.fixed)
mc.data = data_calibration
mc.fixed = fixed_calibration
run_model!(mc)
rebuild_model!(mc)
mc.fixed = deepcopy(fixed_default)
run_model!(mc)   # base solve; mc.data now holds the solved variables

function dumpkey(io, prefix, dict, k)
    v = dict[k]
    if isa(v, Number)
        println(io, "$k,$v")
    else
        na = v
        for idx in CartesianIndices(size(na))
            val = na[idx]
            key = join([string(names(na, d)[idx[d]]) for d in 1:ndims(na)], ",")
            println(io, "$k,$key,$val")
        end
    end
end

open(out_csv, "w") do io
    for k in sort(collect(keys(mc.data)))
        try; dumpkey(io, "d", mc.data, k); catch e; println(">>> skip data $k: $e"); end
    end
    for k in sort(collect(keys(mc.parameters)))
        try; dumpkey(io, "p", mc.parameters, k); catch e; println(">>> skip par $k: $e"); end
    end
    # sets too, so the port can align indices
    for k in sort(collect(keys(mc.sets)))
        try
            for (i, e) in enumerate(mc.sets[k])
                println(io, "SET_$k,$i,$(string(e))")
            end
        catch e; println(">>> skip set $k: $e"); end
    end
end
println(">>> DONE")
