# Dump Julia's SHOCKED full solution (all mc.data vars after calibrate -> base ->
# tariff shock). Args: <dataset> <tariff_power> <out_csv>. The shock oracle for
# the Julia-vs-equilibria parity tool.

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

dataset      = length(ARGS) >= 1 ? ARGS[1] : "sample"
tariff_power = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.10
out_csv      = length(ARGS) >= 3 ? ARGS[3] : "julia_shocksol.csv"

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
sd = deepcopy(mc.data)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, sd)
fd = deepcopy(mc.fixed)
mc.data = data_calibration
mc.fixed = fixed_calibration
run_model!(mc)
rebuild_model!(mc)
mc.fixed = deepcopy(fd)
run_model!(mc)  # base

# MULTIPLICATIVE tariff shock (× base power), matching GEMPACK, the levels block,
# run_julia_oracle.jl and the Pyomo port. A flat `= tariff_power` under-shocks
# positive-tariff routes and biases the Armington sourcing response.
for c in hSets["comm"], s in hSets["reg"], d in hSets["reg"]
    try; mc.data["tms"][c, s, d] *= tariff_power; catch; end
end
run_model!(mc)  # shock

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
    for k in sort(collect(keys(mc.data)))
        try; dk(io, k); catch e; println(">>> skip $k: $e"); end
    end
end
println(">>> DONE")
