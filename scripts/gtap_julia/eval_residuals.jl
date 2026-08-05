# Evaluate Julia's constraint residuals at an EXTERNAL point (equilibria's shocked
# solution) by calling solve_model with max_iter=0 (IPOPT reports the initial
# constraint violation at the seeded point without moving). The constraint with
# the largest residual is the equation equilibria's root violates — the exact bug.
# Args: <tariff> <point_csv> <out_txt>

const PKG = "/Users/marmol/proyectos/GlobalTradeAnalysisProjectModelV7.jl"
include(joinpath(PKG, "src/GlobalTradeAnalysisProjectModelV7.jl"))
using Main.GlobalTradeAnalysisProjectModelV7
using NamedArrays, JuMP

tariff  = parse(Float64, ARGS[1])
pt_csv  = ARGS[2]
out_txt = ARGS[3]

(; hData, hParameters, hSets) = get_sample_data()
mc = generate_initial_model(hSets=hSets, hData=hData, hParameters=hParameters)
sd = deepcopy(mc.data)
(; fixed_calibration, data_calibration) = generate_calibration_inputs(mc, sd)
fd = deepcopy(mc.fixed)
mc.data = data_calibration; mc.fixed = fixed_calibration
run_model!(mc)
rebuild_model!(mc)
mc.fixed = deepcopy(fd)

# external point
pt = Dict{String, Dict{Tuple, Float64}}()
for line in eachline(pt_csv)
    (startswith(line, ">>>") || isempty(line)) && continue
    parts = split(line, ","); name = parts[1]; val = tryparse(Float64, parts[end])
    val === nothing && continue
    idx = Tuple(parts[2:end-1])
    haskey(pt, name) || (pt[name] = Dict{Tuple, Float64}())
    pt[name][idx] = val
end
for (name, cells) in pt
    haskey(mc.data, name) || continue
    na = mc.data[name]; isa(na, Number) && continue
    for idx in CartesianIndices(size(na))
        key = Tuple(string(names(na, d)[idx[d]]) for d in 1:ndims(na))
        haskey(cells, key) && (na[idx] = cells[key])
    end
end
for c in hSets["comm"], s in hSets["reg"], d in hSets["reg"]
    try; mc.data["tms"][c, s, d] = tariff; catch; end
end

# solve_model with max_iter=0: builds model seeded at mc.data, IPOPT reports
# initial infeasibility. We then read each constraint residual at that point.
sets = mc.sets; data = mc.data; parameters = mc.parameters; fixed = mc.fixed
(; constraints) = solve_model(; sets, data, parameters, fixed, max_iter=0)

# Re-evaluate: build a fresh model and read constraint primal values at the seed.
# solve_model already set start values = data. Use JuMP's dual/primal? Simpler:
# recompute residual per constraint name from the returned constraint list.
open(out_txt, "w") do io
    worst = Dict{String, Float64}()
    for c in constraints
        r = try
            abs(JuMP.value(c))  # primal value at the (unmoved, max_iter=0) solution
        catch
            0.0
        end
        cn = String(strip(split(split(string(c), "[")[1], ":")[1]))
        worst[cn] = max(get(worst, cn, 0.0), isnan(r) ? 0.0 : r)
    end
    for (cn, r) in sort(collect(worst), by=x->-x[2])[1:min(30, length(worst))]
        println(io, "$cn  $r")
    end
end
println(">>> DONE")
