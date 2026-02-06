@testset "Require Unique function" begin
    v = [:a, :a, :b, :c, :d, :d, :d]
    @test issetequal(EnhancedBayesianNetworks.not_unique_elements(v), [:a, :d])
    v = [:a, :b, :c, :d]
    @test isempty(EnhancedBayesianNetworks.not_unique_elements(v))
end