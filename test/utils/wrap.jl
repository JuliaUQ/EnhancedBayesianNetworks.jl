@testitem "Wrap functions" begin
    x = :a
    @test EnhancedBayesianNetworks.wrap(x) == [x]
    x = [:a]
    @test EnhancedBayesianNetworks.wrap(x) == x
    x = [1 2 3; 3 4 5; 5 6 7]
    @test EnhancedBayesianNetworks.wrap(x) == x
end