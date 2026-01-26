using EnhancedBayesianNetworks

@testset "EnhancedBayesianNetwork" begin
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = 0.5
    weather[:W=>:cloudy] = 0.5

    sprinkler = DiscreteNode(:S, [:W])
    sprinkler[:W=>:sunny, :S=>:on] = 0.7
    sprinkler[:W=>:sunny, :S=>:off] = 0.3
    sprinkler[:W=>:cloudy, :S=>:on] = 0.05
    sprinkler[:W=>:cloudy, :S=>:off] = 0.95

    rain = DiscreteNode(:R, [:W])
    rain[:W=>:sunny, :R=>:yes] = 0.05
    rain[:W=>:sunny, :R=>:no] = 0.95
    rain[:W=>:cloudy, :R=>:yes] = 0.7
    rain[:W=>:cloudy, :R=>:no] = 0.3

    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.1, 0.2)
    grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass[:R=>:no, :S=>:off, :G=>:wet] = 0

    @testset "Structure" begin
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        @test net.A == sparse(zeros(length(nodes), length(nodes)))
        @test net.topology == Dict(:W => 1, :G => 2, :R => 3, :S => 4)
    end

    # @testset "is cyclic dfs & unique names" begin

    #     cpt_root = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
    #     cpt_root[:A=>:a1] = 0.5
    #     cpt_root[:A=>:a2] = 0.5
    #     root = DiscreteNode(:A, cpt_root)

    #     child1 = DiscreteNode(:B, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:D => [:d1, :d2, :d1, :d2, :d1, :d2, :d1, :d2], :A => [:a1, :a1, :a2, :a2, :a1, :a1, :a2, :a2], :B => [:b1, :b2, :b1, :b2, :b1, :b2, :b1, :b2], :Π => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))

    #     child2 = DiscreteNode(:C, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:B => [:b1, :b1, :b2, :b2], :C => [:c1, :c2, :c1, :c2], :Π => [0.5, 0.5, 0.5, 0.5])))

    #     child3 = DiscreteNode(:D, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:C => [:c1, :c1, :c2, :c2], :D => [:d1, :d2, :d1, :d2], :Π => [0.5, 0.5, 0.5, 0.5])))

    #     net = EnhancedBayesianNetwork([root, child1, child2, child3])
    #     add_child!(net, :A, :B)
    #     add_child!(net, :B, :C)
    #     add_child!(net, :C, :D)
    #     add_child!(net, :D, :B)
    #     @test EnhancedBayesianNetworks._is_cyclic_dfs(net.A)
    #     @test_throws ErrorException("network is cyclic!") order!(net)

    #     grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Π => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    #     grass = DiscreteNode(:g, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(grass_states))

    #     nodes = [weather, sprinkler, rain, grass]
    #     net = EnhancedBayesianNetwork(nodes)
    #     add_child!(net, :w, :s)
    #     add_child!(net, :w, :r)
    #     add_child!(net, :s, :g)
    #     add_child!(net, :r, :g)

    #     order!(net)
    #     @test net.A == sparse(Matrix([0 1.0 1.0 0; 0 0 0 1.0; 0 0 0 1.0; 0 0 0 0]))
    #     @test net.topology == Dict(:w => 1, :s => 2, :g => 4, :r => 3)
    #     @test net.nodes == [weather, sprinkler, rain, grass]
    #     @test isnothing(EnhancedBayesianNetworks._verify_net(net))

    #     rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :g => [:no_rain, :rain, :no_rain, :rain], :Π => [0.9, 0.1, 0.2, 0.8])
    #     rain = DiscreteNode(:g, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state))

    #     nodes = [weather, sprinkler, rain, grass]
    #     @test_throws ErrorException("network nodes names must be unique") EnhancedBayesianNetwork(nodes)

    #     rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:on, :off, :on, :off], :Π => [0.9, 0.1, 0.2, 0.8])
    #     rain = DiscreteNode(:r, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state))

    #     nodes = [weather, sprinkler, rain, grass]
    #     @test_throws ErrorException("network nodes states must be unique") EnhancedBayesianNetwork(nodes)
    # end

    # @testset "Markov Blankets" begin
    #     x1 = DiscreteNode(:x1, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:x1 => [:x1y, :x1n], :Π => [0.5, 0.5])))
    #     x2 = DiscreteNode(:x2, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:x2 => [:x2y, :x2n], :Π => [0.5, 0.5])))
    #     x4 = DiscreteNode(:x4, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:x4 => [:x4y, :x4n], :Π => [0.5, 0.5])))
    #     x8 = DiscreteNode(:x8, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:x8 => [:x8y, :x8n], :Π => [0.5, 0.5])))
    #     x3_states = DataFrame(
    #         :x1 => [:x1y, :x1y, :x1n, :x1n],
    #         :x3 => [:x3y, :x3n, :x3y, :x3n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )

    #     x3 = DiscreteNode(:x3, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x3_states))
    #     x5_states = DataFrame(
    #         :x2 => [:x2y, :x2y, :x2n, :x2n],
    #         :x5 => [:x5y, :x5n, :x5y, :x5n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )
    #     x5 = DiscreteNode(:x5, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x5_states))
    #     x7_states = DataFrame(
    #         :x4 => [:x4y, :x4y, :x4n, :x4n],
    #         :x7 => [:x7y, :x7n, :x7y, :x7n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )
    #     x7 = DiscreteNode(:x7, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x7_states))
    #     x11_states = DataFrame(
    #         :x8 => [:x8y, :x8y, :x8n, :x8n],
    #         :x11 => [:x11y, :x11n, :x11y, :x11n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )
    #     x11 = DiscreteNode(:x11, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x11_states))
    #     x6_states = DataFrame(
    #         :x4 => [:x4y, :x4y, :x4y, :x4y, :x4n, :x4n, :x4n, :x4n],
    #         :x3 => [:x3y, :x3y, :x3n, :x3n, :x3y, :x3y, :x3n, :x3n],
    #         :x6 => [:x6y, :x6n, :x6y, :x6n, :x6y, :x6n, :x6y, :x6n],
    #         :Π => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #     )
    #     x6 = DiscreteNode(:x6, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x6_states))
    #     x9_states = DataFrame(
    #         :x6 => [:x6y, :x6y, :x6y, :x6y, :x6n, :x6n, :x6n, :x6n],
    #         :x5 => [:x5y, :x5y, :x5n, :x5n, :x5y, :x5y, :x5n, :x5n],
    #         :x9 => [:x9y, :x9n, :x9y, :x9n, :x9y, :x9n, :x9y, :x9n],
    #         :Π => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #     )
    #     x9 = DiscreteNode(:x9, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x9_states))
    #     x10_states = DataFrame(
    #         :x6 => [:x6y, :x6y, :x6y, :x6y, :x6n, :x6n, :x6n, :x6n],
    #         :x8 => [:x8y, :x8y, :x8n, :x8n, :x8y, :x8y, :x8n, :x8n],
    #         :x10 => [:x10y, :x10n, :x10y, :x10n, :x10y, :x10n, :x10y, :x10n],
    #         :Π => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    #     )
    #     x10 = DiscreteNode(:x10, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x10_states))
    #     x12_states = DataFrame(
    #         :x9 => [:x9y, :x9y, :x9n, :x9n],
    #         :x12 => [:x12y, :x12n, :x12y, :x12n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )
    #     x12 = DiscreteNode(:x12, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x12_states))
    #     x13_states = DataFrame(
    #         :x10 => [:x10y, :x10y, :x10n, :x10n],
    #         :x13 => [:x13y, :x13n, :x13y, :x13n],
    #         :Π => [0.5, 0.5, 0.5, 0.5]
    #     )
    #     x13 = DiscreteNode(:x13, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(x13_states))
    #     nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
    #     ebn = EnhancedBayesianNetwork(nodes)
    #     add_child!(ebn, :x1, :x3)
    #     add_child!(ebn, :x2, :x5)
    #     add_child!(ebn, :x4, :x7)
    #     add_child!(ebn, :x8, :x11)
    #     add_child!(ebn, :x3, :x6)
    #     add_child!(ebn, :x4, :x6)
    #     add_child!(ebn, :x5, :x9)
    #     add_child!(ebn, :x6, :x9)
    #     add_child!(ebn, :x6, :x10)
    #     add_child!(ebn, :x8, :x10)
    #     add_child!(ebn, :x9, :x12)
    #     add_child!(ebn, :x10, :x13)
    #     order!(ebn)

    #     markov_bl = markov_blanket(ebn, 9)
    #     @test issetequal(markov_bl[1], [5 10 4 11 3 8])
    #     @test issetequal(markov_bl[2], [:x3, :x4, :x5, :x8, :x9, :x10])
    #     @test issetequal(markov_bl[3], [x3, x4, x5, x8, x9, x10])
    #     @test markov_blanket(ebn, :x6) == markov_bl
    #     @test markov_blanket(ebn, x6) == markov_bl
    # end

    # @testset "Markov Envelopes" begin
    #     Y1 = DiscreteNode(:y1, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:y1 => [:yy1, :yn1], :Π => [0.5, 0.5])), Dict(:yy1 => [Parameter(0.5, :y1)], :yn1 => [Parameter(0.8, :y1)]))
    #     X1 = ContinuousNode(:x1, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Normal())))
    #     X2 = ContinuousNode(:x2, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Normal())))
    #     X3 = ContinuousNode(:x3, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Normal())))

    #     model = Model(df -> df.y1 .+ df.x1, :y2)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     performance = df -> df.y2
    #     Y2 = DiscreteFunctionalNode(:y2, models, performance, simulation)

    #     model = Model(df -> df.x1 .+ df.x2, :y3)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     performance = df -> df.y3
    #     Y3 = DiscreteFunctionalNode(:y3, models, performance, simulation)

    #     model = Model(df -> df.x3 .+ df.x2, :y4)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     performance = df -> df.y4
    #     Y4 = DiscreteFunctionalNode(:y4, models, performance, simulation)

    #     model = Model(df -> df.x3, :y5)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     performance = df -> df.y5
    #     parameter = Dict(:fail_y5 => [Parameter(1, :y5)], :fail_y5 => [Parameter(0, :y5)])
    #     Y5 = DiscreteFunctionalNode(:y5, models, performance, simulation, parameter)

    #     model = Model(df -> df.y3, :x4)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     X4 = ContinuousFunctionalNode(:x4, models, simulation)

    #     model = Model(df -> df.x4, :y6)
    #     models = [model]
    #     simulation = MonteCarlo(200)
    #     performance = df -> df.y6
    #     Y6 = DiscreteFunctionalNode(:y6, models, performance, simulation)
    #     nodes = [X1, X2, X3, Y1, Y2, Y3, Y4, Y5, X4, Y6]
    #     ebn = EnhancedBayesianNetwork(nodes)

    #     add_child!(ebn, :y1, :y2)
    #     add_child!(ebn, :x1, :y2)
    #     add_child!(ebn, :x1, :y3)
    #     add_child!(ebn, :x2, :y3)
    #     add_child!(ebn, :x2, :y4)
    #     add_child!(ebn, :x3, :y4)
    #     add_child!(ebn, :x3, :y5)
    #     add_child!(ebn, :y5, :x4)
    #     add_child!(ebn, :x4, :y6)
    #     @suppress order!(ebn)

    #     @test issetequal(EnhancedBayesianNetworks._get_markov_group(ebn, Y5), [Y5, X4, X3])
    #     envelopes = markov_envelope(ebn)
    #     @test issetequal(envelopes[1], [X1, X2, X3, Y1, Y2, Y3, Y4, Y5])
    #     @test issetequal(envelopes[2], [Y6, Y5, X4])
    # end
end