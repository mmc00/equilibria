# Run the Julia GTAPv7 model on ONE OF OUR aggregated datasets (basedata.har /
# default.prm / sets.har) — no aggregation (our data is post-GTAPAgg). Calibrate,
# solve base, apply the tariff shock, dump base + shock full solutions.
#
# Args: <dataset_dir> <tariff_power> <out_base_csv> <out_shock_csv>
#   dataset_dir holds basedata.har, default.prm, sets.har.

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays
import HeaderArrayFile

ddir      = ARGS[1]
tariff    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.10
out_base  = length(ARGS) >= 3 ? ARGS[3] : "our_base.csv"
out_shock = length(ARGS) >= 4 ? ARGS[4] : "our_shock.csv"

# Read our HARs (already aggregated) into NamedArray dicts, lower-casing headers to
# the names the model expects.
function readdict(path)
    h = HeaderArrayFile.File(path)
    Dict(lowercase(String(k)) => NamedArray(v) for (k, v) in h)
end

println(">>> reading our HARs"); flush(stdout)
hData       = readdict(joinpath(ddir, "basedata.har"))
hParameters = readdict(joinpath(ddir, "default.prm"))
hSets       = readdict(joinpath(ddir, "sets.har"))

println(">>> generate_initial_model"); flush(stdout)
mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
start_data = deepcopy(mc.data)

println(">>> calibrate"); flush(stdout)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, start_data)
fixed_default = deepcopy(mc.fixed)
mc.data = data_calibration; mc.fixed = fixed_calibration
run_model!(mc)
rebuild_model!(mc)
mc.fixed = deepcopy(fixed_default)

regs = hSets["reg"]; comms = hSets["comm"]

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
function dumpall(path)
    open(path, "w") do io
        for k in sort(collect(keys(mc.data))); try; dk(io, k); catch; end; end
        for k in sort(collect(keys(mc.sets)))
            try; for (i, e) in enumerate(mc.sets[k]); println(io, "SET_$k,$i,$(string(e))"); end; catch; end
        end
    end
end

println(">>> base solve"); flush(stdout)
run_model!(mc)
dumpall(out_base)

println(">>> shock (tms *= $tariff)"); flush(stdout)
# MULTIPLICATIVE tariff shock (× base power), matching GEMPACK `Shock tm=uniform 10`,
# the levels block ((1+imptx)*1.10), run_from_csv.jl, run_julia_oracle.jl and the
# Pyomo port. A flat `= tariff` under-shocks positive-tariff routes.
for c in comms, s in regs, d in regs
    try; mc.data["tms"][c, s, d] *= tariff; catch; end
end
run_model!(mc)
dumpall(out_shock)

println(">>> DONE"); flush(stdout)
