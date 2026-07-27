@testitem "Show - _pq_string" setup = [SetupShowNet] begin
    @test _pq_string([:X], EnhancedBayesianNetworks.Evidence()) == "P(X)"
    @test _pq_string([:X], EnhancedBayesianNetworks.Evidence(:V => :YesV)) == "P(X | V=YesV)"
    @test _pq_string([:X, :Y], EnhancedBayesianNetworks.Evidence()) == "P(X, Y)"
end

@testitem "Show - Posterior" setup = [SetupShowNet] begin
    @test sprint(show, post) == "Posterior P(W)"
    @test sprint(show, poste) == "Posterior P(W | X=on)"
    p = plainshow(post)
    @test occursin("Posterior P(W)", p)
    @test occursin("Probability", p)
    @test occursin("Cloudy", p)
    @test occursin("Sunny", p)
end

@testitem "Show - CredalPosterior" setup = [SetupShowNet] begin
    @test sprint(show, cpost) == "CredalPosterior P(W)"
    p = plainshow(cpost)
    @test occursin("CredalPosterior P(W)", p)
    @test occursin("Interval", p)
    @test occursin("Extreme posteriors: 1", p)
end