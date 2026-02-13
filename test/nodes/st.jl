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

        st = SimulationTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x, :y])
        st[:x=>:x1, :y=>:y1] = MonteCarlo(10)
        st[:x=>:x1, :y=>:y2] = MonteCarlo(20)
        st[:x=>:x1, :y=>:y3] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))
        st[:x=>:x2, :y=>:y1] = MonteCarlo(40)
        st[:x=>:x2, :y=>:y2] = DoubleLoop(MonteCarlo(50))
        st[:x=>:x2, :y=>:y3] = MonteCarlo(60)
        filtering1 = filter(st, ([:x => :x1])...)
        @test isa(filtering1, SubDataFrame)
        @test issetequal(filtering1.sim, [MonteCarlo(10), MonteCarlo(20), SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))])
        filtering2 = filter(st, ([:x, :y] .=> [:x2, :y2])...)
        @test isa(filtering2, SubDataFrame)
        @test issetequal(filtering2.sim, [DoubleLoop(MonteCarlo(50))])
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

        st = SimulationTable{EnhancedBayesianNetworks.ContinuousSimulation}([:x, :y])
        st[:x=>:x1, :y=>:y1] = MonteCarlo(10)
        st[:x=>:x1, :y=>:y2] = MonteCarlo(20)
        st[:x=>:x1, :y=>:y3] = MonteCarlo(30)
        st[:x=>:x2, :y=>:y1] = MonteCarlo(40)
        st[:x=>:x2, :y=>:y2] = MonteCarlo(50)
        st[:x=>:x2, :y=>:y3] = MonteCarlo(60)
        filtering1 = filter(st, ([:x => :x1])...)
        @test isa(filtering1, SubDataFrame)
        @test issetequal(filtering1.sim, [MonteCarlo(10), MonteCarlo(20), MonteCarlo(30)])
        filtering2 = filter(st, ([:x, :y] .=> [:x2, :y2])...)
        @test isa(filtering2, SubDataFrame)
        @test issetequal(filtering2.sim, [MonteCarlo(50)])
    end
end