# Run the Julia GTAPv7 model on one of OUR aggregated datasets, fed via the flat
# CSVs from export_har_for_julia.py (data.csv / params.csv / sets.csv). Rebuilds
# the (hData, hParameters, hSets) NamedArrays with the exact GTAP header dimensions
# (extracted from Julia's own sample), calibrates, solves base + tariff shock, and
# dumps both full solutions. Bypasses HeaderArrayFile (our HAR metadata differs).
#
# Args: <csv_dir> <tariff_power> <out_base_csv> <out_shock_csv>

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays

csvdir    = ARGS[1]
tariff    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.10
out_base  = length(ARGS) >= 3 ? ARGS[3] : "our_base.csv"
out_shock = length(ARGS) >= 4 ? ARGS[4] : "our_shock.csv"

# header -> tuple of set names (from Julia's sample; commodity axes named "acts").
const DDIM = Dict(
    "evfb"=>["endw","acts","reg"], "evfp"=>["endw","acts","reg"], "evos"=>["endw","acts","reg"],
    "makb"=>["acts","acts","reg"], "maks"=>["acts","acts","reg"],
    "pop"=>["reg"], "save"=>["reg"], "vdep"=>["reg"], "vkb"=>["reg"],
    "vcif"=>["acts","reg","reg"], "vfob"=>["acts","reg","reg"], "vmsb"=>["acts","reg","reg"], "vxsb"=>["acts","reg","reg"],
    "vdfb"=>["acts","acts","reg"], "vdfp"=>["acts","acts","reg"], "vmfb"=>["acts","acts","reg"], "vmfp"=>["acts","acts","reg"],
    "vdgb"=>["acts","reg"], "vdgp"=>["acts","reg"], "vdib"=>["acts","reg"], "vdip"=>["acts","reg"],
    "vdpb"=>["acts","reg"], "vdpp"=>["acts","reg"], "vmgb"=>["acts","reg"], "vmgp"=>["acts","reg"],
    "vmib"=>["acts","reg"], "vmip"=>["acts","reg"], "vmpb"=>["acts","reg"], "vmpp"=>["acts","reg"],
    "vst"=>["marg","reg"], "vtwr"=>["marg","acts","reg","reg"],
)
const PDIM = Dict(
    "esbc"=>["acts","reg"], "esbd"=>["acts","reg"], "esbm"=>["acts","reg"], "esbq"=>["acts","reg"],
    "esbt"=>["acts","reg"], "esbv"=>["acts","reg"], "etrq"=>["acts","reg"], "incp"=>["acts","reg"], "subp"=>["acts","reg"],
    "esbg"=>["reg"], "rflx"=>["reg"], "etre"=>["endw","reg"], "esbs"=>["marg"],
)

# read sets.csv -> Dict{setname => Vector{String}}. Lower-case members: Julia
# hardcodes endwc=["capital"] (prepare_sets.jl:5), and raw GTAP codes are lower —
# our sets use "Capital"/"Land"/... so we lower-case to match Julia's expectations.
setmem = Dict{String, Vector{String}}()
for line in eachline(joinpath(csvdir, "sets.csv"))
    isempty(line) && continue
    p = split(line, ","); s = p[1]; m = lowercase(String(p[end]))
    haskey(setmem, s) || (setmem[s] = String[])
    push!(setmem[s], m)
end
# GTAP: commodity axis uses acts members (comm==acts here)
setmem["actsdim"] = setmem["acts"]

axes_for(dims) = Tuple(setmem[d == "acts" ? "acts" : d] for d in dims)

# read a values CSV -> Dict{header => flat Dict{tuple(Int) => value}}
function readvals(path)
    out = Dict{String, Dict{Vector{Int}, Float64}}()
    for line in eachline(path)
        isempty(line) && continue
        p = split(line, ",")
        h = String(p[1])
        v = tryparse(Float64, p[end]); v === nothing && continue
        idx = [parse(Int, x) + 1 for x in p[2:end-1]]   # python 0-based -> julia 1-based
        haskey(out, h) || (out[h] = Dict{Vector{Int}, Float64}())
        out[h][idx] = v
    end
    out
end

function build(vals, dimmap)
    d = Dict{String, Any}()
    for (h, cells) in vals
        if !haskey(dimmap, h)
            # scalar (e.g. rdlt) or flag table (eflg) — handle scalars
            if length(first(keys(cells))) == 0
                d[h] = first(values(cells))
            end
            continue
        end
        dims = dimmap[h]
        ax = axes_for(dims)
        arr = fill(NaN, map(length, ax)...)
        for (idx, v) in cells
            arr[idx...] = v
        end
        d[h] = NamedArray(arr, ax)
    end
    d
end

println(">>> reading CSVs"); flush(stdout)
dvals = readvals(joinpath(csvdir, "data.csv"))
pvals = readvals(joinpath(csvdir, "params.csv"))

hData = build(dvals, DDIM)
hParameters = build(pvals, PDIM)

# eflg: endw x 3-flag table (sluggish/mobile/fixed). Build as NamedArray(endw, [sluggish,mobile,fixed]).
if haskey(pvals, "eflg")
    # eflg column order matches Julia's sample: [mobile, sluggish, fixed]. Our HAR
    # flag table (dumped as-is by export_har_for_julia.py) uses the same column order,
    # so land lands in "sluggish" and labor/capital in "mobile". Getting this wrong
    # inverts endws↔endwm → γ_qes2 stores the (Inf) mobile rows and drops finite land.
    endw = setmem["endw"]; flags = ["mobile", "sluggish", "fixed"]
    arr = zeros(length(endw), length(flags))
    for (idx, v) in pvals["eflg"]; arr[idx...] = v; end
    hParameters["eflg"] = NamedArray(arr, (endw, flags))
end
# rdlt scalar (capFlex = 1)
haskey(pvals, "rdlt") && (hParameters["rdlt"] = first(values(pvals["rdlt"])))

hSets = Dict("reg"=>setmem["reg"], "comm"=>setmem["comm"], "acts"=>setmem["acts"],
             "endw"=>setmem["endw"], "marg"=>setmem["marg"])

println(">>> generate_initial_model"); flush(stdout)
mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
start_data = deepcopy(mc.data)
println(">>> calibrate"); flush(stdout)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, start_data)
fd = deepcopy(mc.fixed); mc.data = data_calibration; mc.fixed = fixed_calibration
run_model!(mc); rebuild_model!(mc); mc.fixed = deepcopy(fd)

regs = setmem["reg"]; comms = setmem["comm"]
function dk(io, k)
    v = mc.data[k]
    if isa(v, Number); println(io, "$k,$v")
    else
        for idx in CartesianIndices(size(v))
            key = join([string(names(v, d)[idx[d]]) for d in 1:ndims(v)], ",")
            println(io, "$k,$key,$(v[idx])")
        end
    end
end
# dump elasticity/structural parameters too (esubt/esubc/esubm/etrae/...), so the Pyomo
# port (gtap_julia) can seed itself from this CSV without a separate Julia HAR dump —
# the port's _get(sol,...) reads params and shares from one flat namespace.
function dkp(io, k)
    v = mc.parameters[k]
    if isa(v, Number); println(io, "$k,$v")
    else
        try
            for idx in CartesianIndices(size(v))
                key = join([string(names(v, d)[idx[d]]) for d in 1:ndims(v)], ",")
                println(io, "$k,$key,$(v[idx])")
            end
        catch; end
    end
end
dumpall(path) = open(path, "w") do io
    for k in sort(collect(keys(mc.data))); try; dk(io, k); catch; end; end
    for k in sort(collect(keys(mc.parameters))); try; dkp(io, k); catch; end; end
    for k in sort(collect(keys(mc.sets))); try; for (i,e) in enumerate(mc.sets[k]); println(io,"SET_$k,$i,$(string(e))"); end; catch; end; end
end

println(">>> base solve"); flush(stdout)
run_model!(mc); dumpall(out_base)
println(">>> shock"); flush(stdout)
# Import-tariff shock. The GEMPACK .cmf says `Shock tm = uniform 10`, but the block
# model (99.5% vs the fixture) implements it as a ×1.10 MULTIPLICATIVE shock to the
# BILATERAL power imptx[rp,i,r] (≈ tms), and crucially recovers the extra revenue in
# regional income from that SAME bilateral wedge. In the Julia model, e_y collects
# import-tariff revenue only from the tms term (qxs·pcif·(tms-1)) — NOT from tm — so
# shocking tm leaves the revenue un-collected → regional income (and all final demand)
# over-contracts 2-3×. Shock tms MULTIPLICATIVELY (×tariff, preserving base bilateral
# rates) so the price wedge AND the income revenue both move consistently.
tp = tariff  # e.g. 1.10 = +10%
# Shock ALL bilateral tms including the diagonal: in this log-value model the self-trade
# self-tms IS part of the equilibrium (EU_28/ROW have real self-trade), and skipping it
# breaks the solution (drops to 40%). GEMPACK's `Shock tm = uniform 10` hits every route.
for c in comms, s in regs, d in regs
    try; mc.data["tms"][c,s,d] = mc.data["tms"][c,s,d] * tp; catch; end
end
run_model!(mc); dumpall(out_shock)
println(">>> DONE"); flush(stdout)
