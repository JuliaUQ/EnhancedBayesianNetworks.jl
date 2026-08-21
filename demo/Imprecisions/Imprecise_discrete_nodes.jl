using EnhancedBayesianNetworks

W = DiscreteNode(:W)
W[:W => :sunny] = 0.5
W[:W => :cloudy] = 0.5

S = DiscreteNode(:S, [:W])
S[:W => :sunny, :S => :on] = Interval(0.8, 0.95)   # imprecise entries
S[:W => :sunny, :S => :off] = Interval(0.05, 0.2)
S[:W => :cloudy, :S => :on] = 0.2
S[:W => :cloudy, :S => :off] = 0.8

cn = CredalNetwork([W, S])
add_child!(cn, W, S)
order!(cn)

gplot(cn, background_color = "white", legend = true, label_size = 10, legend_x = 15, legend_y = 14)

infer(cn, :S, Evidence())

M = 3.2633           # kg/cm/s²  (sprung mass)
m = 0.8158           # kg/cm/s²  (unsprung mass)
g = 981              # cm/s²     (gravity)

A = DiscreteNode(:A, [:road => [Parameter(0.15915, :A)], :offroad => [Parameter(0.8, :A)]])  # A in rad·cm²/m
A[:A => :road] = 0.7
A[:A => :offroad] = 0.3

b₀ = DiscreteNode(:b₀, [:normal_load => [Parameter(0.27, :b₀)], :over_load => [Parameter(0.5, :b₀)]])
b₀[:b₀ => :normal_load] = Interval(0.4, 0.6)
b₀[:b₀ => :over_load] = Interval(0.4, 0.6)

V = ContinuousNode(:V, Uniform(7, 12))            # m/s       (vehicle velocity)
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

E = DiscreteFunctionalNode(:E, [model], performance, MonteCarlo(10^4))

ebn = EnhancedBayesianNetwork([A, b₀, V, C, Cₖ, K, E])
add_child!(ebn, [A, b₀, V, C, Cₖ, K], E)
order!(ebn)

gplot(ebn, background_color = "white", legend = true, label_size = 10, legend_x = 15, legend_y = 14)

elapsed = @elapsed cn = reduce(ebn)
println("network reduced in ", round(elapsed; digits = 3), " s")

gplot(cn, background_color = "white", node_scale = 1.1, title = "Reduced Credal Network", label_size = 12)

infer(cn, :E, Evidence())

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
