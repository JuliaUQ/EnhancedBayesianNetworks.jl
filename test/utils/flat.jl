@testset "Flat function" begin
    x = 0.6
    @test EnhancedBayesianNetworks.flat(x) == [0.6, 0.6]
    x = Interval(0.1, 0.3)
    @test EnhancedBayesianNetworks.flat(x) == [0.1, 0.3]
end