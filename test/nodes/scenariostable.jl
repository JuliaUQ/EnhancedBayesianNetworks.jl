@testitem "Discrete CPT" setup=[ExtraDeps] begin
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x], :Π)
    @test names(st.data) == ["x", "Π"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}(:x, :Π)
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y], :Π)
    st[:x=>:yesx, :y=>:yesy] = 0.1
    st[:x=>:yesx, :y=>:noy] = 0.9
    st[:x=>:nox, :y=>:yesy] = 0.2
    st[:x=>:nox, :y=>:noy] = 0.8
    @test st.data.Π == [0.1, 0.9, 0.2, 0.8]
    @test st[:x=>:yesx, :y=>:yesy] == 0.1
    @test st[:x=>:yesx, :y=>:noy] == 0.9
    @test st[:x=>:nox, :y=>:yesy] == 0.2
    @test st[:x=>:nox, :y=>:noy] == 0.8

    @test_throws ErrorException("Cannot set index with [:x] into a ScenariosTable initialized with [:x, :y]") st[:x=>:x1] = 0.3
    @test_throws ErrorException("Cannot set index with [:x, :y, :z] into a ScenariosTable initialized with [:x, :y]") st[:x=>:yesx, :y=>:yesy, :z=>:z1] = 0.3

    push!(st.data, (x=:yesx, y=:yesy, Π=0.5))
    @test_throws AssertionError st[:x=>:yesx, :y=>:yesy] = 0.3
    @test_throws AssertionError st[:x=>:yesx, :y=>:yesy]

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y], :Π)
    st[:x=>:yesx, :y=>:yesy] = Interval(0.1, 0.2)
    st[:x=>:yesx, :y=>:noy] = 0.9
    st[:x=>:nox, :y=>:yesy] = 0.2
    st[:x=>:nox, :y=>:noy] = 0.8
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
    @test st.data.Π == [Interval(0.1, 0.2), 0.9, 0.2, 0.8]
    @test st[:x=>:yesx, :y=>:yesy] == Interval(0.1, 0.2)
    @test st[:x=>:yesx, :y=>:noy] == 0.9
    @test st[:x=>:nox, :y=>:yesy] == 0.2
    @test st[:x=>:nox, :y=>:noy] == 0.8

    @test_throws ErrorException("Index [:x => :x1] not found in the ScenariosTable $st") st[:x=>:x1]
    @test_throws ErrorException("Index [:x => :x1, :y => :y3] not found in the ScenariosTable $st") st[:x=>:x1, :y=>:y3]

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x], :Π)
    @test_throws ArgumentError("Probability -0.2 must be >= 0 and <= 1") st[:x=>:yesx] = -0.2
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x], :Π)
    @test_throws ArgumentError("Probability 2 must be >= 0 and <= 1") st[:x=>:yesx] = 2
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x], :Π)
    @test_throws ArgumentError("Probability [0.1, 1.1] must be >= 0 and <= 1") st[:x=>:yesx] = Interval(0.1, 1.1)
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x], :Π)
    @test_throws ArgumentError("Probability [-0.1, 0.9] must be >= 0 and <= 1") st[:x=>:yesx] = Interval(-0.1, 0.9)

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y], :Π)
    st[:x=>:x1, :y=>:y1] = 0.1
    st[:x=>:x1, :y=>:y2] = 0.2
    st[:x=>:x1, :y=>:y3] = 0.7
    st[:x=>:x2, :y=>:y1] = 0.01
    st[:x=>:x2, :y=>:y2] = 0.09
    st[:x=>:x2, :y=>:y3] = 0.9
    filtering1 = filter(st, ([:x => :x1])...)
    @test isa(filtering1, SubDataFrame)
    @test issetequal(filtering1.Π, [0.1, 0.2, 0.7])
    filtering2 = filter(st, ([:x, :y] .=> [:x2, :y2])...)
    @test isa(filtering2, SubDataFrame)
    @test issetequal(filtering2.Π, [0.09])
end

@testitem "Continuous CPT" setup=[ExtraDeps] begin
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousProbability}(:x, :Π)
    @test names(st.data) == ["x", "Π"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousProbability}([:x], :Π)
    @test names(st.data) == ["x", "Π"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousProbability}([:x], :Π)
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
    st[:x=>:yesx] = Normal()
    st[:x=>:nox] = Normal(2, 1)
    @test st.data.Π == [Normal(), Normal(2, 1)]
    @test st[:x=>:yesx] == Normal()
    @test st[:x=>:nox] == Normal(2, 1)

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousProbability}(Symbol[], :Π)
    @test names(st.data) == ["Π"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
    @test eltype(st.data.Π) == EnhancedBayesianNetworks.ContinuousProbability

    st[] = Normal()
    @test st.data.Π == [Normal()]
    @test st[] == Normal()

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousProbability}([:x, :y], :Π)
    st[:x=>:x1, :y=>:y1] = Normal(0, 1)
    st[:x=>:x1, :y=>:y2] = Normal(1, 1)
    st[:x=>:x2, :y=>:y1] = Normal(-1, 1)
    st[:x=>:x2, :y=>:y2] = Normal(-2, 1)

    @test_throws ErrorException("Cannot set index with [:x] into a ScenariosTable initialized with [:x, :y]") st[:x=>:x1] = Normal()
    @test_throws ErrorException("Cannot set index with [:x, :y, :z] into a ScenariosTable initialized with [:x, :y]") st[:x=>:x1, :y=>:y1, :z=>:z1] = Normal()

    @test_throws ErrorException("Index [:x => :x3] not found in the ScenariosTable $st") st[:x=>:x3]
    @test_throws ErrorException("Index [:x => :x1, :y => :y3] not found in the ScenariosTable $st") st[:x=>:x1, :y=>:y3]

    filtering1 = filter(st, ([:x => :x1])...)
    @test isa(filtering1, SubDataFrame)
    @test issetequal(filtering1.Π, [Normal(0, 1), Normal(1, 1)])
    filtering2 = filter(st, ([:x, :y] .=> [:x1, :y2])...)
    @test isa(filtering2, SubDataFrame)
    @test issetequal(filtering2.Π, [Normal(1, 1)])
end

@testitem "Discrete ST" setup=[ExtraDeps] begin
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x], :sim)
    @test names(st.data) == ["x", "sim"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteSimulation
    @test eltype(st.data.sim) == EnhancedBayesianNetworks.DiscreteSimulation
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteSimulation}(:x, :sim)
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.DiscreteSimulation
    @test eltype(st.data.sim) == EnhancedBayesianNetworks.DiscreteSimulation
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x, :y], :sim)
    st[:x=>:yesx, :y=>:yesy] = MonteCarlo(100)
    st[:x=>:yesx, :y=>:noy] = DoubleLoop(MonteCarlo(100))
    st[:x=>:nox, :y=>:yesy] = RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
    st[:x=>:nox, :y=>:noy] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

    @test st.data.sim == [MonteCarlo(100), DoubleLoop(MonteCarlo(100)), RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))), SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))]
    @test st[:x=>:yesx, :y=>:yesy] == MonteCarlo(100)
    @test st[:x=>:yesx, :y=>:noy] == DoubleLoop(MonteCarlo(100))
    @test st[:x=>:nox, :y=>:yesy] == RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
    @test st[:x=>:nox, :y=>:noy] == SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

    @test_throws ErrorException("Cannot set index with [:x] into a ScenariosTable initialized with [:x, :y]") st[:x=>:x1] = MonteCarlo(20)
    @test_throws ErrorException("Cannot set index with [:x, :y, :z] into a ScenariosTable initialized with [:x, :y]") st[:x=>:x1, :y=>:y1, :z=>:z1] = MonteCarlo(20)
    @test_throws ErrorException("Index [:x => :maybex] not found in the ScenariosTable $st") st[:x=>:maybex]

    push!(st.data, (x=:yesx, y=:yesy, sim=MonteCarlo(10)))
    @test_throws AssertionError st[:x=>:yesx, :y=>:yesy] = MonteCarlo(10)
    @test_throws AssertionError st[:x=>:yesx, :y=>:yesy]

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteSimulation}([:x, :y], :sim)
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

@testitem "Continuous ST" setup=[ExtraDeps] begin
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousSimulation}(:x, :sim)
    @test names(st.data) == ["x", "sim"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousSimulation
    @test eltype(st.data.sim) == EnhancedBayesianNetworks.ContinuousSimulation
    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousSimulation}([:x], :sim)
    @test names(st.data) == ["x", "sim"]
    @test typeof(st).parameters[1] == EnhancedBayesianNetworks.ContinuousSimulation
    @test eltype(st.data.sim) == EnhancedBayesianNetworks.ContinuousSimulation
    st[:x=>:yesx] = MonteCarlo(20)
    st[:x=>:nox] = MonteCarlo(10)
    @test st.data.sim == [MonteCarlo(20), MonteCarlo(10)]
    @test st[:x=>:yesx] == MonteCarlo(20)
    @test st[:x=>:nox] == MonteCarlo(10)
    @test_throws MethodError st[:x=>:nox] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))

    @test_throws ErrorException("Cannot set index with [:y] into a ScenariosTable initialized with [:x]") st[:y=>:x1] = MonteCarlo(20)
    @test_throws ErrorException("Cannot set index with [:x, :y] into a ScenariosTable initialized with [:x]") st[:x=>:x1, :y=>:y1] = MonteCarlo(20)

    @test_throws ErrorException("Index [:x => :maybex] not found in the ScenariosTable $st") st[:x=>:maybex]

    push!(st.data, (x=:yesx, sim=MonteCarlo(10)))
    @test_throws AssertionError st[:x=>:yesx] = MonteCarlo(10)
    @test_throws AssertionError st[:x=>:yesx]

    st = EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.ContinuousSimulation}([:x, :y], :sim)
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

@testitem "verify probability values" begin
    @test_throws ArgumentError EnhancedBayesianNetworks.verify_probability_value(1.1)
    @test_throws ArgumentError EnhancedBayesianNetworks.verify_probability_value(Interval(0.5, 1.2))
end