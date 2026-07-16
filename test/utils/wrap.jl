@testitem "Wrap functions" begin
    x = :a
    @test EnhancedBayesianNetworks._wrap(x) == [x]
    x = [:a]
    @test EnhancedBayesianNetworks._wrap(x) == x
end