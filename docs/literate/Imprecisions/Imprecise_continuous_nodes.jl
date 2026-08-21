# # Imprecision on a Continuous Node
#
# The [vehicle suspension example](../EnhancedBayesianNetworks/vehicle-3D-suspension.md)
# treats every input as precise. Here we revisit the same benchmark of Gerasimov &
# Vořechovský [gerasimov_failure_2023](@cite) — same model, loads and coefficients —
# but make the vehicle speed `V` **imprecise**: instead of a distribution, each road
# condition fixes `V` only to an interval (a p-box with unknown shape). This example
# shows how EnhancedBayesianNetworks propagates that imprecision, and — importantly —
# which simulation each modelling choice requires.
#
# Two choices decide how the imprecision reaches the failure node and how it must be
# simulated:
#
# 1. whether `V` carries a **discretization**, and
# 2. whether the functional node is **discrete** (a reliability analysis) or
#    **continuous** (a distribution/p-box that is reconstructed).
#
# The four cases below walk through the combinations.

using EnhancedBayesianNetworks

# ## The shared model
#
# The masses, gravity, road/load scenario nodes, the three precise suspension
# coefficients and the composite limit state are exactly those of the precise
# example; only `V` will change from case to case, so we define everything else once.

M = 3.2633           # kg/cm/s²  (sprung mass)
m = 0.8158           # kg/cm/s²  (unsprung mass)
g = 981              # cm/s²     (gravity)

A = DiscreteNode(:A, [:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]])  # A in rad·cm²/m
A[:A=>:road] = 0.7
A[:A=>:offroad] = 0.3

b₀ = DiscreteNode(:b₀, [:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]])
b₀[:b₀=>:normal_load] = 0.7
b₀[:b₀=>:over_load] = 0.3

C = ContinuousNode(:C, Normal(431.7221, 10))     # kg/cm    (suspension stiffness)
Cₖ = ContinuousNode(:Cₖ, Normal(1475.5503, 10))   # kg/cm    (tire stiffness)
K = ContinuousNode(:K, Normal(55.0406, 10))       # kg/cm/s  (damping coefficient)

function composite_model(A, b₀, V, M, m, g, C, Cₖ, K)
    g1 = 1 .- (π .* m .* V .* A) ./ (b₀ .* K .* g .^ 2) .* [(Cₖ ./ (m .+ M) .- (C ./ M)) .^ 2 .+ C .^ 2 ./ (m .* M) .+ Cₖ .* K .^ 2 ./ (m .* M .^ 2)]
    g2 = 4000 .* C .* (M .* g) .^ (-1.5) .- 8.6394
    g3 = 2 .* .√(M .* g .* (K .^ 2 .* Cₖ ./ (C .* (m .+ M)) .+ C)) .- 1
    g4 = Cₖ .- [g .* (M .+ m)] .^ 0.877
    return minimum([g1[1], g2, g3, g4[1]])
end

model = Model(df -> composite_model.(df.A, df.b₀, df.V, M, m, g, df.C, df.Cₖ, df.K), :y)
performance = df -> df.y

# `V` is imprecise: on a `road` its speed lies in `[11, 13]` m/s, off-road in
# `[6, 8]` m/s, with no assumed distribution inside those bands.

v_road = Interval(11, 13)      # m/s
v_offroad = Interval(6, 8)     # m/s

# ## Case 1 — imprecise `V`, no discretization, feeding a discrete functional node: double-loop
#
# When `V` is *not* discretized it reaches the failure node `E` as an imprecise
# continuous input. The reliability analysis at `E` therefore has to explore that
# imprecision, which is exactly what a `DoubleLoop` simulation does: an outer
# loop over the interval and an inner Monte-Carlo loop. The reduced network is credal,
# and `E`'s own conditional probability table is **interval-valued** — a lower and
# upper failure probability per scenario.

V = ContinuousNode(:V, [:A])
V[:A=>:road] = v_road
V[:A=>:offroad] = v_offroad

E = DiscreteFunctionalNode(:E, [model], performance, DoubleLoop(MonteCarlo(10^3)))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, A, V)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

#-

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits=3), " s")

# The failure node's table now carries intervals — the bounds on the collapse
# probability for each road/load scenario:

cn.nodes[findfirst(n -> n.name == :E, cn.nodes)].cpt.data

#-

gplot(cn, background_color="white", node_scale=1.1, title="Reduced Credal Network", label_size=12)


# ## Case 2 — imprecise `V`, discretized, feeding a discrete functional node: single-loop
#
# Attaching an [`ApproximatedDiscretization`](@ref) to `V` changes the picture. During
# reduction an imprecise continuous node is **split**: its imprecision moves into the
# discrete surrogate `V_d` (whose bin probabilities become intervals — a *credal*
# node), while the continuous residual handed to `E` is approximated by a **precise**
# distribution. So `E` no longer sees any imprecise continuous input.

discretization_v = ApproximatedDiscretization([6.0, 8.0, 10.0, 13.0], 2)  # edges in m/s

V = ContinuousNode(:V, [:A], discretization_v)
V[:A=>:road] = v_road
V[:A=>:offroad] = v_offroad

E = DiscreteFunctionalNode(:E, [model], performance, DoubleLoop(MonteCarlo(10^6)))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, A, V)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

#-

# Because `E`'s inputs are now precise, a `DoubleLoop` has nothing imprecise to loop
# over and the reduction raises an error. We catch it here to show the message:

try
    reduce(ebn)
catch e
    showerror(stdout, e)
end

# !!! note "Single-loop after discretizing an imprecise child node"
#     When the only imprecision feeding a functional node comes from a non-root imprecise
#     continuous ancestors that are **discretized**, use a **single-loop** simulation
#     (e.g. `MonteCarlo`). Discretization has already carried the imprecision
#     into the credal discrete surrogate; the residual reaching the functional node is
#     precise, so a double loop is neither needed nor valid.
#
# Swapping the simulation to a single-loop `MonteCarlo` fixes it. The imprecision is
# not lost — it now lives in the credal `V_d`, so the reduced network is still a
# `CredalNetwork`:

V = ContinuousNode(:V, [:A], discretization_v)
V[:A=>:road] = v_road
V[:A=>:offroad] = v_offroad

E = DiscreteFunctionalNode(:E, [model], performance, MonteCarlo(10^6))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, A, V)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits=3), " s")

#-

gplot(cn, background_color="white", node_scale=1.1, title="Reduced Credal Network", label_size=12)

# ## Case 3 — imprecise `V`, no discretization, feeding a continuous functional node: MonteCarlo
#
# When the functional node is *continuous* rather than discrete, imprecision is
# handled differently again. Leaving `V` imprecise (no discretization), a
# [`ContinuousFunctionalNode`](@ref) propagates it with an ordinary
# `MonteCarlo` and reconstructs the output as a **probability box** — a
# lower/upper distribution pair. Continuous functional nodes have no double-loop
# variant, so a single-loop `MonteCarlo` is the right (and only) choice: this is the
# continuous counterpart of Case 1.

V = ContinuousNode(:V, [:A])
V[:A=>:road] = v_road
V[:A=>:offroad] = v_offroad

E = ContinuousFunctionalNode(:E, [model], MonteCarlo(10^6))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, A, V)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

#-

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits=3), " s")

#-

gplot(cn, background_color="white", node_scale=1.1, title="Reduced Credal Network", label_size=12)

# The failure node is now a continuous node whose entries are `:lb`/`:ub`
# `EmpiricalDistribution` pairs — the reconstructed p-box:

cn.nodes[findfirst(n -> n.name == :E, cn.nodes)].cpt.data

# ## Case 4 — imprecise `V`, discretized, feeding a continuous functional node: MonteCarlo
#
# The final combination discretizes `V` *and* feeds a continuous functional node —
# the continuous counterpart of Case 2. Discretization again splits `V`: the
# imprecision moves into the credal `V_d`, and the residual reaching `E` is precise.
# So the continuous functional node has only precise inputs and reconstructs a single
# **precise** distribution (not a p-box); a single-loop `MonteCarlo` is all that is
# needed, and the imprecision survives in `V_d`.

V = ContinuousNode(:V, [:A], discretization_v)
V[:A=>:road] = v_road
V[:A=>:offroad] = v_offroad

E = ContinuousFunctionalNode(:E, [model], MonteCarlo(500))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, A, V)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

#-

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits=3), " s")

#-

gplot(cn, background_color="white", node_scale=1.1, title="Reduced Credal Network", label_size=12)

# Unlike Case 3, `E` is now a single precise distribution; the imprecision instead
# lives in the discretized node `V_d`, whose bin probabilities are intervals:

cn.nodes[findfirst(n -> n.name == :V_d, cn.nodes)].cpt.data
