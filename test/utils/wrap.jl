@testitem "Wrap functions" begin
    x = :a
    @test EnhancedBayesianNetworks.wrap(x) == [x]
    x = [:a]
    @test EnhancedBayesianNetworks.wrap(x) == x
end