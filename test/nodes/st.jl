@testset "Simulation Tables" begin
    @testset "Discrete ST" begin
        st = SimulationTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x])
        @test names(st.data) == ["x", "sim"]
        @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteSimulation
        @test eltype(st.data.sim) == EnhancedBayesianNetworks.DiscreteSimulation
        st = SimulationTable{EnhancedBayesianNetworks.DiscreteSimulation}(:x)
        @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteSimulation
        @test eltype(st.data.sim) == EnhancedBayesianNetworks.DiscreteSimulation
        st = SimulationTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x, :y])
        st[:x=>:yesx, :y=>:yesy] = MonteCarlo(100)
        st[:x=>:yesx, :y=>:noy] = DoubleLoop(MonteCarlo(100))
        st[:x=>:nox, :y=>:yesy] = RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
        st[:x=>:nox, :y=>:noy] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

        @test st.data.sim == [MonteCarlo(100), DoubleLoop(MonteCarlo(100)), RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))), SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))]
        @test st[:x=>:yesx, :y=>:yesy] == MonteCarlo(100)
        @test st[:x=>:yesx, :y=>:noy] == DoubleLoop(MonteCarlo(100))
        @test st[:x=>:nox, :y=>:yesy] == RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
        @test st[:x=>:nox, :y=>:noy] == SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

        @test_throws ErrorException("Cannot set index with [:x] into a SimulationTable initialized with [:x, :y]") st[:x=>:x1] = MonteCarlo(20)
        @test_throws ErrorException("Cannot set index with [:x, :y, :z] into a SimulationTable initialized with [:x, :y]") st[:x=>:x1, :y=>:y1, :z=>:z1] = MonteCarlo(20)

        @test_throws ErrorException("index not find in the SimlationTable $st") st[:x=>:maybex]

        push!(st.data, (x=:yesx, y=:yesy, sim=MonteCarlo(10)))
        @test_throws AssertionError st[:x=>:yesx, :y=>:yesy] = MonteCarlo(10)
        @test_throws AssertionError st[:x=>:yesx, :y=>:yesy]
    end

    @testset "Continuous ST" begin
        st = SimulationTable{EnhancedBayesianNetworks.ContinuousSimulation}(:x)
        @test names(st.data) == ["x", "sim"]
        @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousSimulation
        @test eltype(st.data.sim) == EnhancedBayesianNetworks.ContinuousSimulation
        st = SimulationTable{EnhancedBayesianNetworks.ContinuousSimulation}([:x])
        @test names(st.data) == ["x", "sim"]
        @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousSimulation
        @test eltype(st.data.sim) == EnhancedBayesianNetworks.ContinuousSimulation
        st[:x=>:yesx] = MonteCarlo(20)
        st[:x=>:nox] = MonteCarlo(10)
        @test st.data.sim == [MonteCarlo(20), MonteCarlo(10)]
        @test st[:x=>:yesx] == MonteCarlo(20)
        @test st[:x=>:nox] == MonteCarlo(10)
        @test_throws MethodError st[:x=>:nox] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

        @test_throws ErrorException("Cannot set index with [:y] into a SimulationTable initialized with [:x]") st[:y=>:x1] = MonteCarlo(20)
        @test_throws ErrorException("Cannot set index with [:x, :y] into a SimulationTable initialized with [:x]") st[:x=>:x1, :y=>:y1] = MonteCarlo(20)

        @test_throws ErrorException("index not find in the SimlationTable $st") st[:x=>:maybex]

        push!(st.data, (x=:yesx, sim=MonteCarlo(10)))
        @test_throws AssertionError st[:x=>:yesx] = MonteCarlo(10)
        @test_throws AssertionError st[:x=>:yesx]
    end
end