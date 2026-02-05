@testset "CPTs" begin
    @testset "Discrete CPT" begin
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(:x)
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y])
        cpt[:x=>:yesx, :y=>:yesy] = 0.1
        cpt[:x=>:yesx, :y=>:noy] = 0.9
        cpt[:x=>:nox, :y=>:yesy] = 0.2
        cpt[:x=>:nox, :y=>:noy] = 0.8
        @test cpt.data.Π == [0.1, 0.9, 0.2, 0.8]
        @test cpt[:x=>:yesx, :y=>:yesy] == 0.1
        @test cpt[:x=>:yesx, :y=>:noy] == 0.9
        @test cpt[:x=>:nox, :y=>:yesy] == 0.2
        @test cpt[:x=>:nox, :y=>:noy] == 0.8

        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y])
        cpt[:x=>:yesx, :y=>:yesy] = Interval(0.1, 0.2)
        cpt[:x=>:yesx, :y=>:noy] = 0.9
        cpt[:x=>:nox, :y=>:yesy] = 0.2
        cpt[:x=>:nox, :y=>:noy] = 0.8
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
        @test cpt.data.Π == [Interval(0.1, 0.2), 0.9, 0.2, 0.8]
        @test cpt[:x=>:yesx, :y=>:yesy] == Interval(0.1, 0.2)
        @test cpt[:x=>:yesx, :y=>:noy] == 0.9
        @test cpt[:x=>:nox, :y=>:yesy] == 0.2
        @test cpt[:x=>:nox, :y=>:noy] == 0.8

        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test_throws ErrorException("provided probability value -0.2 is unfeasible") cpt[:x=>:yesx] = -0.2
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test_throws ErrorException("provided probability value 2 is unfeasible") cpt[:x=>:yesx] = 2
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test_throws ErrorException("provided probability value [0.1, 1.1] is unfeasible") cpt[:x=>:yesx] = Interval(0.1, 1.1)
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test_throws ErrorException("provided probability value [-0.1, 0.9] is unfeasible") cpt[:x=>:yesx] = Interval(-0.1, 0.9)

        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y])
        cpt[:x=>:x1, :y=>:y1] = 0.1
        cpt[:x=>:x1, :y=>:y2] = 0.2
        cpt[:x=>:x1, :y=>:y3] = 0.7
        cpt[:x=>:x2, :y=>:y1] = 0.01
        cpt[:x=>:x2, :y=>:y2] = 0.09
        cpt[:x=>:x2, :y=>:y3] = 0.9
        filtering1 = filter(cpt, ([:x => :x1])...)
        @test isa(filtering1, SubDataFrame)
        @test issetequal(filtering1.Π, [0.1, 0.2, 0.7])
        filtering2 = filter(cpt, ([:x, :y] .=> [:x2, :y2])...)
        @test isa(filtering2, SubDataFrame)
        @test issetequal(filtering2.Π, [0.09])
    end

    @testset "Continuous CPT" begin
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}(:x)
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}([:x])
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt[:x=>:yesx] = Normal()
        cpt[:x=>:nox] = Normal(2, 1)

        @test cpt.data.Π == [Normal(), Normal(2, 1)]

        @test cpt[:x=>:yesx] == Normal()
        @test cpt[:x=>:nox] == Normal(2, 1)

        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}(Symbol[])
        @test names(cpt.data) == ["Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability

        cpt[] = Normal()
        @test cpt.data.Π == [Normal()]
        @test cpt[] == Normal()

        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}([:x, :y])
        cpt[:x=>:x1, :y=>:y1] = Normal(0, 1)
        cpt[:x=>:x1, :y=>:y2] = Normal(1, 1)
        cpt[:x=>:x2, :y=>:y1] = Normal(-1, 1)
        cpt[:x=>:x2, :y=>:y2] = Normal(-2, 1)
        filtering1 = filter(cpt, ([:x => :x1])...)
        @test isa(filtering1, SubDataFrame)
        @test issetequal(filtering1.Π, [Normal(0, 1), Normal(1, 1)])
        filtering2 = filter(cpt, ([:x, :y] .=> [:x1, :y2])...)
        @test isa(filtering2, SubDataFrame)
        @test issetequal(filtering2.Π, [Normal(1, 1)])
    end

    @testset "CPT from DataFrame" begin
        df = DataFrame(:a => [:a1, :a2], :Π => [:a, 0])
        @test_throws AssertionError ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(df)
        df = DataFrame(:a => [:a1, :a2], :c => [1, 0])
        @test_throws AssertionError ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(df)
        df = DataFrame(:a => [:a1, :a2], :Π => [1, 0])
        cpt = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(df)
        @test cpt[:a=>:a1] == 1
        @test cpt[:a=>:a2] == 0
    end
end