@testitem "Verify add_child!" begin
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = 0.5
    weather[:W=>:cloudy] = 0.5

    sprinkler_parameters = [:on => [Parameter(1, :S)], :off => [Parameter(0, :S)]]
    sprinkler = DiscreteNode(:S, [:W], sprinkler_parameters)
    sprinkler[:W=>:sunny, :S=>:on] = 0.7
    sprinkler[:W=>:sunny, :S=>:off] = 0.3
    sprinkler[:W=>:cloudy, :S=>:on] = 0.05
    sprinkler[:W=>:cloudy, :S=>:off] = 0.95

    rain = DiscreteNode(:R, [:W])
    rain[:W=>:sunny, :R=>:yes] = 0.05
    rain[:W=>:sunny, :R=>:no] = 0.95
    rain[:W=>:cloudy, :R=>:yes] = 0.7
    rain[:W=>:cloudy, :R=>:no] = 0.3

    storm = ContinuousNode(:St, [:W])
    storm[:W=>:sunny] = Normal()
    storm[:W=>:cloudy] = Normal(2, 1)

    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.1, 0.2)
    grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass[:R=>:no, :S=>:off, :G=>:wet] = 0

    simulation = MonteCarlo(100)
    model_t = Model(df -> df.S .+ df.St, :T)
    performance_t = df -> df.T
    tree = DiscreteFunctionalNode(:Tr, model_t, performance_t, simulation)

    model_b = Model(df -> df.S .+ df.St, :B)
    bush = ContinuousFunctionalNode(:B, model_b, simulation)

    nodes = [weather, grass, rain, sprinkler, storm, bush, tree]
    net = EnhancedBayesianNetwork(nodes)
    par = weather
    ch = [sprinkler, storm, grass]
    @test_throws ErrorException("Invalid Network: node :G does not have the node :W in its CPT") EnhancedBayesianNetworks._verify_discrete(par, ch)

    par = storm
    ch = [sprinkler]
    @test_throws ErrorException("Invalid Network: nodes [:S] are not functional nodes and cannot be children of the continuous/functional node :St") EnhancedBayesianNetworks._verify_continuous_and_functional(par, ch)

    par = bush
    ch = [sprinkler]
    @test_throws ErrorException("Invalid Network: nodes [:S] are not functional nodes and cannot be children of the continuous/functional node :B") EnhancedBayesianNetworks._verify_continuous_and_functional(par, ch)
end