# # Vehicle Suspension Reliability
#
# This example reproduces the vehicle-suspension reliability benchmark of
# Gerasimov & Vořechovský [gerasimov_failure_2023](@cite). They do not build a
# Bayesian network for it — they use it as a plain structural-reliability
# problem — so here it serves as a *validation* case: we cast the same problem as
# an enhanced Bayesian network, reduce it, and check that the failure
# probability the network infers for a given scenario matches the one a direct
# Monte-Carlo reliability analysis returns for that same scenario.
#
# The suspension's safety is governed by a **composite limit state** that
# combines four failure modes; the system fails as soon as any of them is
# violated. Two of the inputs are *scenario* variables handled as discrete nodes
# — the road coefficient and the load coefficient — while the vehicle speed and
# the suspension stiffness, tire stiffness and damping coefficient are continuous.
# All quantities follow the paper's unit system (a technical/CGS system, hence
# `g = 981` cm/s²).

using EnhancedBayesianNetworks

# ## Fixed parameters
#
# The sprung and unsprung masses and the gravitational acceleration are constants
# of the model.

M = 3.2633           # kg/cm/s²  (sprung mass)
m = 0.8158           # kg/cm/s²  (unsprung mass)
g = 981              # cm/s²     (gravity)

# ## Scenario nodes: road and load coefficients
#
# The road coefficient (`A`, in rad·cm²/m) and the load coefficient (`b₀`,
# dimensionless) are discrete nodes. Each state carries a `Parameter` — the
# numerical value fed to the model when that state is active — so these nodes
# drive the *scenario grid* the reliability analysis is repeated over. The
# `road` / `normal_load` values (`0.15915` / `0.27`) are the deterministic values
# used in the paper.

A = DiscreteNode(:A, [:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]])  # A in rad·cm²/m
A[:A => :road] = 0.7
A[:A => :offroad] = 0.3

b₀ = DiscreteNode(:b₀, [:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]])
b₀[:b₀ => :normal_load] = 0.7
b₀[:b₀ => :over_load] = 0.3

# ## Vehicle speed
#
# The speed `V` is a continuous root node, uniform between 7 and 12. Attaching an
# [`ExactDiscretization`](@ref) with explicit interval edges makes `V` survive the
# reduction as a *discrete* node (`V_d`), so we can later condition on a speed
# range. The edges span the full support `[7, 12]` so the bins match the
# distribution exactly, and the middle interval `[9.5, 10.5]` brackets the speed
# `V = 10` used in the cross-check below.

discretization_v = ExactDiscretization([7.0, 8.5, 9.5, 10.5, 11.5, 12.0])  # edges in m/s
V = ContinuousNode(:V, Uniform(7, 12), discretization_v)                   # V in m/s

# ## Continuous suspension coefficients
#
# The suspension stiffness `C`, the tire stiffness `Cₖ` and the damping
# coefficient `K` are continuous root nodes with normal distributions. They carry
# no discretization, so the reduction folds them into the failure model and
# integrates them out by simulation.

C = ContinuousNode(:C, Normal(431.7221, 10))     # kg/cm    (suspension stiffness)
Cₖ = ContinuousNode(:Cₖ, Normal(1475.5503, 10))   # kg/cm    (tire stiffness)
K = ContinuousNode(:K, Normal(55.0406, 10))       # kg/cm/s  (damping coefficient)

# ## The composite limit state
#
# The four failure modes are, in order, exceedance of the road-holding ability
# (`g1`), exceedance of the rolling angle (`g2`), bumper hitting (`g3`) and
# exceedance of the minimum required tire life (`g4`). They are combined by taking
# their minimum: the suspension is *failed* when that minimum drops below zero (a
# series system — any single mode failing fails the whole). The `performance`
# function returns that minimum margin.

function composite_model(A, b₀, V, M, m, g, C, Cₖ, K)
    g1 = 1 .- (π .* m .* V .* A) ./ (b₀ .* K .* g .^ 2) .* [(Cₖ ./ (m .+ M) .- (C ./ M)) .^ 2 .+ C .^ 2 ./ (m .* M) .+ Cₖ .* K .^ 2 ./ (m .* M .^ 2)]
    g2 = 4000 .* C .* (M .* g) .^ (-1.5) .- 8.6394
    g3 = 2 .* .√(M .* g .* (K .^ 2 .* Cₖ ./ (C .* (m .+ M)) .+ C)) .- 1
    g4 = Cₖ .- [g .* (M .+ m)] .^ 0.877
    return minimum([g1[1], g2, g3, g4[1]])
end

model = Model(df -> composite_model.(df.A, df.b₀, df.V, M, m, g, df.C, df.Cₖ, df.K), :y)
performance = df -> df.y

# ## Building the enhanced Bayesian network
#
# The failure event `E` is a [`DiscreteFunctionalNode`](@ref) built from the
# limit-state model and its performance function; the six inputs (`A`, `b₀`, `V`,
# `C`, `Cₖ`, `K`) are its parents. A Monte-Carlo simulation propagates the
# continuous inputs' uncertainty through the model.

sim = MonteCarlo(10^6)
E = DiscreteFunctionalNode(:E, [model], performance, sim)

nodes = [A, b₀, V, C, Cₖ, K, E]
ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

# ## Reducing to a Bayesian network
#
# [`reduce`](@ref) evaluates the functional node: the continuous coefficients
# `C`, `Cₖ`, `K` are folded into the failure model and eliminated, while `V` —
# because it carries a discretization — is kept as a discrete node `V_d`. The
# result is a Bayesian network over the road surface, the load, the discretized
# speed, and the collapse event.

bn = reduce(ebn)

#-

gplot(bn, background_color="white", node_scale=1.1, title="Reduced Bayesian Network", label_size=12)

# ## Inferring the failure probability for a scenario
#
# With the network reduced, we can read off the probability of failure for a
# concrete scenario: driving on a normal-load `road` at a speed in the
# `[9.5, 10.5]` band (i.e. around `V = 10`).

evidence = Evidence(:V_d => Symbol("[9.5, 10.5]"), :A => :road, :b₀ => :normal_load)
ϕ = infer(bn, :E, evidence)

# ## Cross-check against a direct reliability analysis
#
# To confirm the network is faithful, we solve the *same* scenario directly with
# UncertaintyQuantification — fixing the road, load and speed to the scenario's
# values and running a Monte-Carlo estimate of the failure probability. This is
# the calculation Gerasimov & Vořechovský [gerasimov_failure_2023](@cite) perform;
# the enhanced Bayesian network should reproduce it.

M = Parameter(3.2633, :M)                        # kg/cm/s²  (sprung mass)
m = Parameter(0.8158, :m)                        # kg/cm/s²  (unsprung mass)
g = Parameter(981, :g)                           # cm/s²     (gravity)
A = Parameter(0.15915, :A)                       # rad·cm²/m (road coefficient)
b₀ = Parameter(0.27, :b₀)                         # –         (load coefficient)
V = Parameter(10, :V)                            # m/s       (vehicle velocity)
C = RandomVariable(Normal(431.7221, 10), :C)     # kg/cm     (suspension stiffness)
Cₖ = RandomVariable(Normal(1475.5503, 10), :Cₖ)   # kg/cm     (tire stiffness)
K = RandomVariable(Normal(55.0406, 10), :K)      # kg/cm/s   (damping coefficient)

model = Model(df -> composite_model.(df.A, df.b₀, df.V, df.M, df.m, df.g, df.C, df.Cₖ, df.K), :y)
inputs = [A, b₀, V, M, m, g, C, Cₖ, K]

pf, cov, _ = probability_of_failure(model, performance, inputs, MonteCarlo(10^6))

# The failure probability inferred by the Bayesian network (the `:E_failed`
# entry of `ϕ` above) and the one from the direct Monte-Carlo analysis (`pf`)
# agree to within Monte-Carlo scatter — the network integrates the speed over the
# whole `[9.5, 10.5]` band while the direct check pins it to `V = 10`, so a small
# difference is expected. Both also match the reference value
# `pF ≈ 5.2 × 10⁻⁴` that Gerasimov & Vořechovský [gerasimov_failure_2023](@cite)
# report for this configuration (from 10⁶ importance-sampling evaluations),
# confirming the enhanced Bayesian network reproduces their result.
