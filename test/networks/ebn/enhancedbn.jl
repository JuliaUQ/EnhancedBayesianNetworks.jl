@testset "EnhancedBayesianNetwork" begin
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = 0.5
    weather[:W=>:cloudy] = 0.5

    sprinkler_parameter = [:on => [Parameter(0.5, :S)], :off => [Parameter(0, :S)]]
    sprinkler = DiscreteNode(:S, [:W], sprinkler_parameter)
    sprinkler[:W=>:sunny, :S=>:on] = 0.7
    sprinkler[:W=>:sunny, :S=>:off] = 0.3
    sprinkler[:W=>:cloudy, :S=>:on] = 0.05
    sprinkler[:W=>:cloudy, :S=>:off] = 0.95

    rain = DiscreteNode(:R, [:W])
    rain[:W=>:sunny, :R=>:yes] = 0.05
    rain[:W=>:sunny, :R=>:no] = 0.95
    rain[:W=>:cloudy, :R=>:yes] = 0.7
    rain[:W=>:cloudy, :R=>:no] = 0.3

    rain2 = ContinuousNode(:Rc)
    rain2[] = Normal()

    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.1, 0.2)
    grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass[:R=>:no, :S=>:off, :G=>:wet] = 0

    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)

    @testset "Structure" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        @test net.A == sparse(zeros(length(nodes), length(nodes)))
        @test net.topology == Dict(:W => 1, :G => 2, :R => 3, :S => 4, :Rc => 5, :G2 => 6)
        @test net.nodes == nodes
        nodes = [weather, grass, grass, rain, sprinkler, rain2, grass2]
        @test_throws ErrorException("Invalid eBN: duplicate node names [:G]") EnhancedBayesianNetwork(nodes)
        rain3 = DiscreteNode(:R3, [:W])
        rain3[:W=>:sunny, :R3=>:yes] = 0.05
        rain3[:W=>:sunny, :R3=>:maybe] = 0.95
        rain3[:W=>:cloudy, :R3=>:yes] = 0.7
        rain3[:W=>:cloudy, :R3=>:maybe] = 0.3
        nodes = [weather, rain, rain3]
        @test_throws ErrorException("Invalid eBN: duplicate node states [:yes]") EnhancedBayesianNetwork(nodes)
    end

    @testset "add_child!" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        @test_throws ErrorException("Invalid eBN: node(s) '[:W]' have recursion") add_child!(net, weather, weather)
        @test_throws ErrorException("Invalid eBN: node G does not have the node(s) W in its CPT") add_child!(net, weather, grass)
        @test_throws ErrorException("Invalid eBN: node(s) [:G] are not functional node(s) and cannot be children of the continuous/functional node G2") add_child!(net, grass2, grass)
        @test_throws ErrorException("Invalid eBN: node(s) [:G] are not functional node(s) and cannot be children of the continuous/functional node Rc") add_child!(net, rain2, grass)
    end
end