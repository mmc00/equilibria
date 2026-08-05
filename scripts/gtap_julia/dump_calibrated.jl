# Dump Julia's CALIBRATED point (all mc.data keys: α/γ/σ/ϵ params + seeded
# quantities) after calibration, as the reference the port loads directly.
# Args: <dataset> <out_csv>.  CSV rows: key,idx1,idx2,...,value  (scalars: key,value)

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

dataset = length(ARGS) >= 1 ? ARGS[1] : "sample"
out_csv = length(ARGS) >= 2 ? ARGS[2] : "julia_calibrated.csv"

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
mc.data = data_calibration
mc.fixed = fixed_calibration
run_model!(mc)

# After calibration mc.data holds the calibrated params + quantities.
function dumpkey(io, k)
    v = mc.data[k]
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
        try; dumpkey(io, k); catch e; println(">>> skip $k: $e"); end
    end
end
println(">>> DONE")
