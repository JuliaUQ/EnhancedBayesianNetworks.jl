@testitem "Format Intervals" begin
    discretization = ExactDiscretization([-10, 0, 10])
    root = ContinuousNode(:z1, discretization)
    root[] = truncated(Normal(), -10, 10)

    formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
    @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

    discretization = ExactDiscretization([-9, 10])
    root = ContinuousNode(:z1, discretization)
    root[] = truncated(Normal(), -10, 10)

    @test_logs (:warn, "node :z1 has minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

    discretization = ExactDiscretization([-11, 10])
    root = ContinuousNode(:z1, discretization)
    root[] = truncated(Normal(), -10, 10)

    @test_logs (:warn, "node :z1 has minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

    discretization = ExactDiscretization([-10, 9])
    root = ContinuousNode(:z1, discretization)
    root[] = truncated(Normal(), -10, 10)

    @test_logs (:warn, "node :z1 has maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

    discretization = ExactDiscretization([-10, 11])
    root = ContinuousNode(:z1, discretization)
    root[] = truncated(Normal(), -10, 10)

    @test_logs (:warn, "node :z1 has maximum intervals value 11 > support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

    intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
    λ = 2

    exp1 = -Exponential(2) - 1
    exp2 = Exponential(2) + 1
    approx = [
        exp1,
        Uniform(-1, 0),
        Uniform(0.0, 1.0),
        exp2,
    ]
    @test approx == EnhancedBayesianNetworks._approximate.(intervals, λ)
end

@testitem "Discretize Distributions" begin
    # The mass methods read only the p-box settings off the strategy, so the (empty) edges are irrelevant here.
    disc = ExactDiscretization()
    disc_free = ExactDiscretization(; technique = :distribution_free)
    same(a::Interval, b::Interval) = isapprox(a.lb, b.lb; atol = 1.0e-9) && isapprox(a.ub, b.ub; atol = 1.0e-9)

    intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
    dist = Normal()
    probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i, disc), intervals)
    check = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]
    @test all(isapprox.(probs, check, atol = 0.001))

    dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
    probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i, disc), intervals)
    check = [
        Interval(0.022750131948179205, 0.15865525393145702),
        Interval(0.1359051219832778, 0.341344746068543),
        Interval(0.34134474606854304, 0.38292492254802624),
        Interval(0.15865525393145696, 0.5),
    ]
    @test all(same.(probs, check))

    # The distribution-free technique differences the CDF envelopes, which encloses the parametric answer.
    probs_free = map(i -> EnhancedBayesianNetworks._discretize(dist, i, disc_free), intervals)
    @test all(map((f, o) -> f.lb <= o.lb && f.ub >= o.ub, probs_free, probs))
    @test probs_free[3].ub - probs_free[3].lb > probs[3].ub - probs[3].lb
    @test all(map(p -> 0.0 <= p.lb && p.ub <= 1.0, probs_free))

    # Bounds given as a mix of Int and Float: `bounds` then returns a Vector{Real}, which must still reach the optimiser as plain Float64.
    dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 0.5), :σ => 1))
    p = EnhancedBayesianNetworks._discretize(dist, [0.0, 1.0], disc)
    @test same(p, Interval(0.34134474606854304, 0.38292492254802624))

    # A coarser mesh is honoured and still lands close.
    coarse = EnhancedBayesianNetworks._discretize(dist, [0.0, 1.0], ExactDiscretization(; mesh = 1.0e-2))
    @test isapprox(coarse.lb, p.lb; atol = 1.0e-3) && isapprox(coarse.ub, p.ub; atol = 1.0e-3)

    # A parameter given as a degenerate interval leaves nothing to optimise over, and the optimiser cannot be handed a zero-width bound: the mass is then evaluated directly, and a degenerate coordinate alongside a free one is spliced back in rather than searched.
    degenerate = ProbabilityBox{Normal}(Dict(:μ => Interval(0.2, 0.2), :σ => 1))
    @test same(EnhancedBayesianNetworks._discretize(degenerate, [0.0, 1.0], disc), Interval(0.3674043108557064, 0.3674043108557064))
    spliced = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 0.5), :σ => Interval(1.0, 1.0)))
    @test same(EnhancedBayesianNetworks._discretize(spliced, [0.0, 1.0], disc), p)

    intervals = [[-1.0, 0.0], [0.0, 1.0]]
    dist = Interval(-1, 1)
    probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i, disc), intervals)
    check = [Interval(0, 1), Interval(0, 1)]
    @test probs[1] == check[1]
    @test probs[2] == check[2]
    # the technique has no bearing on a precise distribution or an already-vacuous Interval entry
    @test EnhancedBayesianNetworks._discretize(dist, intervals[1], disc_free) == Interval(0, 1)
    @test EnhancedBayesianNetworks._discretize(Normal(), intervals[1], disc_free) == EnhancedBayesianNetworks._discretize(Normal(), intervals[1], disc)
end

@testitem "Approximate" begin
    intervals = [[-Inf, -1.0], [-1.0, 1.0], [1.0, Inf]]
    λ = 2
    approximated_dist = EnhancedBayesianNetworks._approximate.(intervals, λ)
    @test approximated_dist[1].μ == -1
    @test approximated_dist[1].σ == -1
    @test isa(approximated_dist[1].ρ, Exponential)
    @test approximated_dist[1].ρ.θ == λ
    @test approximated_dist[2].a == -1
    @test approximated_dist[2].b == 1
    @test isa(approximated_dist[2], Uniform)
    @test approximated_dist[3].μ == 1
    @test approximated_dist[3].σ == 1
    @test isa(approximated_dist[3].ρ, Exponential)
    @test approximated_dist[3].ρ.θ == λ
end

@testitem "Discretize Nodes" begin
    using Suppressor
    ## Root Nodes
    discretization = ExactDiscretization([-1, 0, 1])
    node = ContinuousNode(:x, discretization)
    node[] = Normal()
    discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
    discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
    @test discretized_node.name == Symbol(string(node.name) * "_d")
    @test isempty(discretized_node.parameters)
    @test isnothing(discretized_node.results)
    @test Symbol.(names(discretized_node.cpt.data)) == [:x_d, :Π]
    @test discretized_node.cpt.data.x_d == discretized_states
    @test isapprox([discretized_node[discretized_node.name .=> i] for i in discretized_states], [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696], atol = 0.001)
    @test new_continuous.name == node.name
    @test isempty(new_continuous.discretization)
    @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
    @test new_continuous.cpt.data.x_d == discretized_states
    @test new_continuous.cpt.data.Π == [
        truncated(Normal(); lower = -Inf, upper = -1.0),
        truncated(Normal(); lower = -1.0, upper = 0.0),
        truncated(Normal(); lower = 0.0, upper = 1.0),
        truncated(Normal(); lower = 1.0, upper = Inf),
    ]

    node = ContinuousNode(:x, discretization)
    node[] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
    discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
    discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
    @test discretized_node.name == Symbol(string(node.name) * "_d")
    @test isempty(discretized_node.parameters)
    @test isnothing(discretized_node.results)
    @test Symbol.(names(discretized_node.cpt.data)) == [:x_d, :Π]
    @test discretized_node.cpt.data.x_d == discretized_states
    masses = [discretized_node[discretized_node.name .=> i] for i in discretized_states]
    check = [
        Interval(0.022750131948179205, 0.15865525393145702),
        Interval(0.1359051219832778, 0.341344746068543),
        Interval(0.34134474606854304, 0.38292492254802624),
        Interval(0.15865525393145696, 0.5),
    ]
    @test all(map((a, b) -> isapprox(a.lb, b.lb; atol = 1.0e-9) && isapprox(a.ub, b.ub; atol = 1.0e-9), masses, check))
    @test new_continuous.name == node.name
    @test isempty(new_continuous.discretization)
    @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
    @test new_continuous.cpt.data.x_d == discretized_states
    @test [new_continuous.cpt.data.Π[i].parameters for i in range(1, 4)] == [
        Dict(:μ => Interval(0, 1), :σ => 1),
        Dict(:μ => Interval(0, 1), :σ => 1),
        Dict(:μ => Interval(0, 1), :σ => 1),
        Dict(:μ => Interval(0, 1), :σ => 1),
    ]
    @test [new_continuous.cpt.data.Π[i].lb for i in range(1, 4)] == [-Inf, -1.0, 0.0, 1.0]
    @test [new_continuous.cpt.data.Π[i].ub for i in range(1, 4)] == [-1.0, 0.0, 1.0, Inf]

    # The technique is read off the node's own strategy, not from a default: the same p-box discretized with :distribution_free gives the enclosure of the masses obtained just above.
    node = ContinuousNode(:x, ExactDiscretization([-1, 0, 1]; technique = :distribution_free))
    node[] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
    free_node, _ = @suppress EnhancedBayesianNetworks._discretize(node)
    free_masses = [free_node[free_node.name .=> i] for i in discretized_states]
    @test all(map((f, o) -> f.lb <= o.lb && f.ub >= o.ub, free_masses, masses))
    @test free_masses[3].ub - free_masses[3].lb > masses[3].ub - masses[3].lb

    ## Child Nodes
    discretization = ApproximatedDiscretization([-1, 0, 1], 2)
    node = ContinuousNode(:x, [:y, :z], discretization)
    node[:y => :y1, :z => :z1] = Normal()
    node[:y => :y1, :z => :z2] = Normal(2, 2)
    node[:y => :y2, :z => :z1] = Interval(0, 3)
    node[:y => :y2, :z => :z2] = Normal(4, 4)
    discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
    discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
    @test discretized_node.name == Symbol(string(node.name) * "_d")
    @test isempty(discretized_node.parameters)
    @test isnothing(discretized_node.results)
    @test Symbol.(names(discretized_node.cpt.data)) == [:y, :z, :x_d, :Π]
    @test unique(discretized_node.cpt.data.x_d) == discretized_states
    @test discretized_node.cpt.data.Π == [
        0.15865525393145702,
        0.06680720126885804,
        Interval(0, 1),
        0.10564977366685525,
        0.341344746068543,
        0.09184805266259898,
        Interval(0, 1),
        0.053005480264601765,
        0.34134474606854304,
        0.14988228479452986,
        Interval(0, 1),
        0.06797209844541116,
        0.15865525393145696,
        0.6914624612740131,
        Interval(0, 1),
        0.7733726476231318,
    ]

    @test new_continuous.name == node.name
    @test isempty(new_continuous.discretization)
    @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
    @test unique(new_continuous.cpt.data.x_d) == discretized_states
    @test isa(new_continuous.cpt.data.Π[1].ρ, Exponential)
    @test isa(new_continuous.cpt.data.Π[2], Uniform)
    @test isa(new_continuous.cpt.data.Π[3], Uniform)
    @test isa(new_continuous.cpt.data.Π[4].ρ, Exponential)
end
