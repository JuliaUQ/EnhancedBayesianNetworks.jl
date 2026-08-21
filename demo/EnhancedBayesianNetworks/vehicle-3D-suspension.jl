using EnhancedBayesianNetworks

M = 3.2633           # kg/cm/s²  (sprung mass)
m = 0.8158           # kg/cm/s²  (unsprung mass)
g = 981              # cm/s²     (gravity)

A = DiscreteNode(:A, [:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]])  # A in rad·cm²/m
A[:A => :road] = 0.7
A[:A => :offroad] = 0.3

b₀ = DiscreteNode(:b₀, [:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]])
b₀[:b₀ => :normal_load] = 0.7
b₀[:b₀ => :over_load] = 0.3

discretization_v = ExactDiscretization([7.0, 8.5, 9.5, 10.5, 11.5, 12.0])  # edges in m/s
V = ContinuousNode(:V, Uniform(7, 12), discretization_v)                   # V in m/s

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

sim = MonteCarlo(10^6)
E = DiscreteFunctionalNode(:E, [model], performance, sim)

nodes = [A, b₀, V, C, Cₖ, K, E]
ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color = "white", legend = true, label_size = 10, legend_x = 15, legend_y = 14)

elapsed = @elapsed bn = reduce(ebn)
println("network reduced in ", round(elapsed; digits = 3), " s")

gplot(bn, background_color = "white", node_scale = 1.1, title = "Reduced Bayesian Network", label_size = 12)

evidence = Evidence(:V_d => Symbol("[9.5, 10.5]"), :A => :road, :b₀ => :normal_load)
elapsed = @elapsed ϕ = infer(bn, :E, evidence)
println("inference completed in ", round(elapsed; digits = 3), " s")
ϕ

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

M = 3.2633           # kg/cm/s²  (sprung mass)
m = 0.8158           # kg/cm/s²  (unsprung mass)
g = 981              # cm/s²     (gravity)

A = DiscreteNode(:A, [:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]])  # A in rad·cm²/m
A[:A => :road] = 0.7
A[:A => :offroad] = 0.3

b₀ = DiscreteNode(:b₀, [:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]])
b₀[:b₀ => :normal_load] = 0.7
b₀[:b₀ => :over_load] = 0.3

discretization_v = ExactDiscretization([7.0, 9.5, 10.5, 12.0])  # edges in m/s (coarser bins)
V = ContinuousNode(:V, Interval(7, 12), discretization_v)       # V in m/s, imprecise

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

sim = DoubleLoop(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
E = DiscreteFunctionalNode(:E, [model], performance, sim)

nodes = [A, b₀, V, C, Cₖ, K, E]
ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color = "white", legend = true, label_size = 10, legend_x = 15, legend_y = 14)

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits = 3), " s")

gplot(cn, background_color = "white", node_scale = 1.1, title = "Reduced Credal Network", label_size = 12)

evidence = Evidence(:V_d => Symbol("[9.5, 10.5]"), :A => :road, :b₀ => :normal_load)
elapsed = @elapsed ϕ = infer(cn, :E, evidence)
println("inference completed in ", round(elapsed; digits = 3), " s")
ϕ

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
