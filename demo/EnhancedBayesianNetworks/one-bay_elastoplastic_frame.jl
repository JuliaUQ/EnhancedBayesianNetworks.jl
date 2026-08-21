using EnhancedBayesianNetworks

μ_gamma = 60         # kN  (mean vertical load V)
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousNode(:V, Gamma(α, θ))

μ_gumbel = 50        # kN  (mean horizontal load H)
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = ContinuousNode(:H, Gumbel(μ_loc, β))

function plastic_moment_capacities(uᵣ)
    ρ = 0.5477
    μ = 150          # kN·m  (mean plastic moment capacity)
    cov = 0.2
    λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)
    normal_μ = λ + ρ * ζ * uᵣ
    normal_std = sqrt((1 - ρ^2) * ζ^2)
    exp(rand(Normal(normal_μ, normal_std)))
end

model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

function frame_model(r1, r2, r3, r4, r5, v, h)
    # moment capacities rᵢ in kN·m, loads v, h in kN; the factor 5 is the
    # member length in m, so 5·(load [kN]) is a moment in kN·m
    g1 = r1 + r2 + r4 + r5 - 5 * h
    g2 = r2 + 2 * r3 + r4 - 5 * v
    g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
    return minimum([g1, g2, g3])
end

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
performance = df -> df.G

Uᵣ = ContinuousNode(:Uᵣ, Normal())
R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(1000))
R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(1000))
R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(1000))
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(1000))
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(1000))

simulation = MonteCarlo(10^6)
frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

bn = reduce(ebn)

gplot(bn, background_color="white", node_scale=1.1, title="Reduced Bayesian Network", label_size=12)

bn.nodes[1].cpt.data

discretizationr4 = ApproximatedDiscretization([20, 49, 51, 149, 151, 220], 1.5)      # edges in kN·m
discretizationr5 = ApproximatedDiscretization([20, 99, 101, 199, 201, 220], 1.5)  # edges in kN·m

Uᵣ = ContinuousNode(:Uᵣ, Normal())
R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(1000))
R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(1000))
R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(1000))
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(1000), discretizationr4)
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(1000), discretizationr5)

simulation = MonteCarlo(10_000)
model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
order!(ebn)

gplot(ebn, background_color="white", legend=true, label_size=10, legend_x=15, legend_y=14)

elapsed = @elapsed bn = reduce(ebn)
println("network reduced in ", round(elapsed; digits=3), " s")

gplot(bn, background_color="white", node_scale=1.1, title="Reduced Bayesian Network", label_size=12)

e1 = Evidence(:R4_d => Symbol("[49.0, 51.0]"), :R5_d => Symbol("[99.0, 101.0]"))
infer(bn, :E, e1)

e2 = Evidence(:R4_d => Symbol("[149.0, 151.0]"), :R5_d => Symbol("[99.0, 101.0]"))
infer(bn, :E, e2)

e3 = Evidence(:R4_d => Symbol("[149.0, 151.0]"), :R5_d => Symbol("[199.0, 201.0]"))
infer(bn, :E, e3)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
