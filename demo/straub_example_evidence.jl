using EnhancedBayesianNetworks
using Distributions
# using Plots
using .MathConstants: γ
using UncertaintyQuantification


n = 10^6
Uᵣ = ContinuousNode{UnivariateDistribution}(:Uᵣ, DataFrame(:Π => Normal()))
μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = ContinuousNode{UnivariateDistribution}(:V, DataFrame(:Π => Gamma(α, θ)))

μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = ContinuousNode{UnivariateDistribution}(:H, DataFrame(:Π => Gumbel(μ_loc, β)))

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

R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
n2 = 2000
discretization1 = ApproximatedDiscretization(collect(range(50, 250, 21)), 1)
discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 21)), 1)
R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n2), discretization1)
R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n2), discretization2)

model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
performance = df -> df.G
simulation = MonteCarlo(n2)
frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

net = EnhancedBayesianNetwork(nodes)

add_child!(net, Uᵣ, R1)
add_child!(net, Uᵣ, R2)
add_child!(net, Uᵣ, R3)
add_child!(net, Uᵣ, R4)
add_child!(net, Uᵣ, R5)
add_child!(net, R1, frame)
add_child!(net, R2, frame)
add_child!(net, R3, frame)
add_child!(net, R4, frame)
add_child!(net, R5, frame)
add_child!(net, V, frame)
add_child!(net, H, frame)
order!(net)

_dispatch!(net)
# reduce!(net)


plt1 = gplot(net)
# savefig(plt1, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Papers/elsarticle-template/imgs/fig7_example_straub_evidence.png")

evidence1 = Dict(
    :R4_d => Symbol([90.0, 100.0]),
    :R5_d => Symbol([100.01, 110.01])
)

ϕ1 = infer(net, :E, evidence1)


evidence2 = Dict(
    :R4_d => Symbol([140.0, 150.0]),
    :R5_d => Symbol([90.01, 100.01])
)

ϕ2 = infer(net, :E, evidence2)

evidence3 = Dict(
    :R4_d => Symbol([150.0, 160.0]),
    :R5_d => Symbol([200.01, 210.01])
)

ϕ3 = infer(net, :E, evidence3)

#### Checking with srps only
### MonteCarlo's Checking
Uᵣ = RandomVariable(Normal(), :Uᵣ)
μ_gamma = 60
cov_gamma = 0.2
α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
V = RandomVariable(Gamma(α, θ), :V)
μ_gumbel = 50
cov_gumbel = 0.4
μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
H = RandomVariable(Gumbel(μ_loc, β), :H,)
model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
# model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
# model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)
r4 = Parameter(50, :R4)
r5 = Parameter(100, :R5)
model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
models = [model1, model2, model3, model]
performance = df -> df.G
inputs = [Uᵣ, V, H, r4, r5]
pf = probability_of_failure(models, performance, inputs, MonteCarlo(10^6))