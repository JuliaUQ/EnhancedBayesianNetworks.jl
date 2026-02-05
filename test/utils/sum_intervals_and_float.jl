@testset "sum intervals and float" begin
    a = Interval(0.2, 0.5)
    b = Interval(0.3, 0.4)
    c = 0.2
    @test EnhancedBayesianNetworks.sum_intervals_and_float(a, b, c) == (0.7, 1.1)
end