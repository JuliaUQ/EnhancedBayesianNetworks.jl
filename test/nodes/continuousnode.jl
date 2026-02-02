@testset "Continuous Nodes" begin
    @testset "structure and setindex" begin
        node_a = ContinuousNode(:a)
        @test isa(node_a, ContinuousNode)
        @test isa(node_a.cpt, ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability})
        @test names(node_a.cpt.data) == ["Π"]
        @test isa(node_a.discretization, ExactDiscretization)
        @test isa(node_a.results, Dict{Vector{Symbol},Tuple})
        node_c = ContinuousNode(:c, [:a])
        @test isa(node_c, ContinuousNode)
        @test isa(node_c.cpt, ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability})
        @test names(node_c.cpt.data) == ["a", "Π"]
        @test isa(node_c.discretization, ApproximatedDiscretization)
        @test isa(node_c.results, Dict{Vector{Symbol},Tuple})
        node_a[] = Normal()
        @test node_a.cpt.data.Π[1] == Normal()
        node_c[:a=>:a1] = Normal()
        node_c[:a=>:a2] = Interval(0.3, 0.6)
        @test node_c.cpt.data.a == [:a1, :a2]
        @test node_c.cpt.data.Π[1] == Normal()
        @test node_c.cpt.data.Π[2] == Interval(0.3, 0.6)
    end
    @testset "main functions" begin
        @test_throws ErrorException(":Π is not allowed as node name") ContinuousNode(:Π)
        node_a = ContinuousNode(:a)
        node_a[] = Normal()
        node_b = ContinuousNode(:b)
        node_b[] = Interval(0.1, 0.3)
        node_c = ContinuousNode(:c, [:a])
        node_c[:a=>:a1] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        node_c[:a=>:a2] = Interval(0.3, 0.6)
        node_d = ContinuousNode(:d, [:a])
        node_d[:a=>:a1] = Normal()
        node_d[:a=>:a2] = Normal(2, 1)

        @test isempty(scenarios(node_a))
        @test isempty(scenarios(node_b))
        @test scenarios(node_c) == [[:a => :a1], [:a => :a2]]
        @test scenarios(node_d) == [[:a => :a1], [:a => :a2]]
        @test isprecise(node_a)
        @test isprecise(node_b) == false
        @test isprecise(node_c) == false
        @test isprecise(node_d)
        @test isroot(node_a)
        @test isroot(node_c) == false
        @test isempty(parents(node_a))
        @test issetequal(parents(node_c), [:a])


        node_e = ContinuousNode(:d, [:a, :b])
        node_e[:a=>:a1, :b=>:b1] = Interval(0.1, 0.2)
        node_e[:a=>:a1, :b=>:b2] = Uniform(0.1, 0.2)
        node_e[:a=>:a2, :b=>:b1] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        node_e[:a=>:a2, :b=>:b2] = Normal()
        evidence = Evidence()
        @test EnhancedBayesianNetworks._inputs(node_a, evidence) == Normal()
        @test EnhancedBayesianNetworks._inputs(node_b, evidence) == Interval(0.1, 0.3)
        @test_throws AssertionError EnhancedBayesianNetworks._inputs(node_c, evidence)

        evidence = Evidence(:a => :a2)
        @test EnhancedBayesianNetworks._inputs(node_a, evidence) == Normal()
        @test EnhancedBayesianNetworks._inputs(node_b, evidence) == Interval(0.1, 0.3)
        @test EnhancedBayesianNetworks._inputs(node_c, evidence) == Interval(0.3, 0.6)
        @test_throws AssertionError EnhancedBayesianNetworks._inputs(node_e, evidence)

        evidence = Evidence(:a => :a1, :b => :b1)
        @test EnhancedBayesianNetworks._inputs(node_a, evidence) == Normal()
        @test EnhancedBayesianNetworks._inputs(node_b, evidence) == Interval(0.1, 0.3)
        @test isa(EnhancedBayesianNetworks._inputs(node_c, evidence), ProbabilityBox)
        @test EnhancedBayesianNetworks._inputs(node_e, evidence) == Interval(0.1, 0.2)
    end
    @testset "auxiliary functions" begin
        dist = Normal()
        @test EnhancedBayesianNetworks._distribution_bounds(dist) == [-Inf, Inf]
        dist = truncated(Normal(), -1, 1)
        @test EnhancedBayesianNetworks._distribution_bounds(dist) == [-1, 1]
        dist = Interval(0, 1)
        @test EnhancedBayesianNetworks._distribution_bounds(dist) == [0, 1]
        dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        @test EnhancedBayesianNetworks._distribution_bounds(dist) == [-Inf, Inf]
        dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1), -1, 1)
        @test EnhancedBayesianNetworks._distribution_bounds(dist) == [-1, 1]
        node_e = ContinuousNode(:d, [:a, :b])
        node_e[:a=>:a1, :b=>:b1] = Interval(0.1, 0.2)
        node_e[:a=>:a1, :b=>:b2] = Uniform(0.1, 0.2)
        node_e[:a=>:a2, :b=>:b1] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        node_e[:a=>:a2, :b=>:b2] = Normal()
        @test EnhancedBayesianNetworks._distribution_bounds(node_e) == [-Inf, Inf]

        dist = Normal()
        @test EnhancedBayesianNetworks._truncate(dist, (0, 1)) == truncated(Normal(), 0, 1)
        dist = Interval(-1, 2)
        @test EnhancedBayesianNetworks._truncate(dist, (0, 1)) == Interval(0, 1)
        dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        res = EnhancedBayesianNetworks._truncate(dist, (0, 1))
        @test res.lb == 0
        @test res.ub == 1
    end
end