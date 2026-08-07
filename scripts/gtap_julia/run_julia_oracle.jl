# Parameterized Julia GTAPv7 oracle: calibrate -> base -> tariff shock -> dump CSV.
# Args: <dataset> <tariff_power> <out_base_csv> <out_shock_csv>
#   dataset: "sample" uses get_sample_data(); otherwise a directory holding
#            gsdfdat.har / gsdfset.har / gsdfpar.har.
# Emits CSV rows: var,idx1,idx2,...,value  for a fixed VAR list, base and shock.

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

dataset      = length(ARGS) >= 1 ? ARGS[1] : "sample"
tariff_power = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.10
out_base     = length(ARGS) >= 3 ? ARGS[3] : "julia_base.csv"
out_shock    = length(ARGS) >= 4 ? ARGS[4] : "julia_shock.csv"

println(">>> dataset=$dataset tariff=$tariff_power"); flush(stdout)

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

println(">>> generate_initial_model"); flush(stdout)
mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
start_data = deepcopy(mc.data)

println(">>> calibrate"); flush(stdout)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, start_data)
fixed_default = deepcopy(mc.fixed)
mc.data = data_calibration
mc.fixed = fixed_calibration
run_model!(mc)

println(">>> rebuild + restore closure"); flush(stdout)
rebuild_model!(mc)
mc.fixed = deepcopy(fixed_default)

regs  = hSets["reg"]
comms = hSets["comm"]

# Dump a NamedArray var from mc.data to CSV rows.
function dumpvar(io, vname)
    try
        na = mc.data[vname]
        for idx in CartesianIndices(size(na))
            v = na[idx]
            key = join([string(names(na, d)[idx[d]]) for d in 1:ndims(na)], ",")
            println(io, "$vname,$key,$v")
        end
    catch e
        println(">>> skip $vname: $e")
    end
end

const VARS = ["qo", "qfe", "qxs", "pds", "qva", "qint", "qc", "pfactor", "qga", "qpa"]

println(">>> base solve"); flush(stdout)
run_model!(mc)
open(out_base, "w") do io
    for v in VARS; dumpvar(io, v); end
end

println(">>> apply tariff shock (tms power *= $tariff_power on all bilateral)"); flush(stdout)
# MULTIPLICATIVE shock: scale each bilateral tms power by tariff_power, matching
# GEMPACK's `Shock tm = uniform 10` (the power rises 10%) and the levels block
# ((1+imptx)*1.10). A route with base power 1.0145 becomes 1.116, NOT a flat 1.10.
# Setting tms = tariff_power (absolute) under-shocks positive-tariff routes and biases
# the Armington sourcing response — it disagreed with run_from_csv.jl (×tariff) and
# with the Pyomo port (model.solve_shock, now × base).
for c in comms, s in regs, d in regs
    try; mc.data["tms"][c, s, d] *= tariff_power; catch; end
end
run_model!(mc)
open(out_shock, "w") do io
    for v in VARS; dumpvar(io, v); end
end

println(">>> DONE"); flush(stdout)
