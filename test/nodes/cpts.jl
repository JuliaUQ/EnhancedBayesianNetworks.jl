@testset "CPTs" begin
    @testset "Discrete CPT" begin
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(:x)
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.DiscreteProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.DiscreteProbability
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:x, :y])
        cpt[:x=>:yesx, :y=>:yesy] = 0.1
        cpt[:x=>:yesx, :y=>:noy] = 0.9
        cpt[:x=>:nox, :y=>:yesy] = 0.2
        cpt[:x=>:nox, :y=>:noy] = 0.8

        @test cpt.data.Π == [0.1, 0.9, 0.2, 0.8]

        @test cpt[:x=>:yesx, :y=>:yesy] == 0.1
        @test cpt[:x=>:yesx, :y=>:noy] == 0.9
        @test cpt[:x=>:nox, :y=>:yesy] == 0.2
        @test cpt[:x=>:nox, :y=>:noy] == 0.8
    end
    @testset "Continuous CPT" begin
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}(:x)
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}([:x])
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability
        cpt[:x=>:yesx] = Normal()
        cpt[:x=>:nox] = Normal(2, 1)

        @test cpt.data.Π == [Normal(), Normal(2, 1)]

        @test cpt[:x=>:yesx] == Normal()
        @test cpt[:x=>:nox] == Normal(2, 1)

        cpt = EnhancedBayesianNetworks.ConditionalProbabilityTable{EnhancedBayesianNetworks.ContinuousProbability}(Symbol[])
        @test names(cpt.data) == ["Π"]
        @test typeof(cpt).parameters[1] == EnhancedBayesianNetworks.ContinuousProbability
        @test eltype(cpt.data.Π) == EnhancedBayesianNetworks.ContinuousProbability

        cpt[] = Normal()
        @test cpt.data.Π == [Normal()]
        @test cpt[] == Normal()
    end
end