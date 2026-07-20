using EnhancedBayesianNetworks
using CairoMakie
using .MathConstants: γ

function plastic_moment_capacities(uᵣ)
    ρ = 0.5477
    μ = 150
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
    g1 = r1 + r2 + r4 + r5 - 5 * h
    g2 = r2 + 2 * r3 + r4 - 5 * v
    g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
    return minimum([g1, g2, g3])
end

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
performance = df -> df.G

n = 10^6
Uᵣ = ContinuousNode(:Uᵣ, Normal())

M = DiscreteNode(:M)
M[:M=>:new] = 0.5
M[:M=>:old] = 0.5

μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousNode(:V, [:M])
V[:M=>:new] = Gamma(α, θ)
V[:M=>:old] = Gamma(α - 1, 2.4)

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = ContinuousNode(:H, Gumbel(μ_loc, β))

R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

simulation = MonteCarlo(n)
frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

parameters_L = [:yesL => [Parameter(1, :L)], :noL => [Parameter(2, :L)]]
L = DiscreteNode(:L, [:M], parameters_L)
L[:M=>:new, :L=>:yesL] = 0.2
L[:M=>:new, :L=>:noL] = 0.8
L[:M=>:old, :L=>:yesL] = 0.5
L[:M=>:old, :L=>:noL] = 0.5

r9 = ContinuousNode(:R9, Normal())

model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
frame2 = DiscreteFunctionalNode(:E2, [model2], df -> df.P, simulation)

nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]
ebn = EnhancedBayesianNetwork(nodes)
add_child!(ebn, M, [V, L])
add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
add_child!(ebn, [L, r9], frame2)
order!(ebn)

gplot(ebn)