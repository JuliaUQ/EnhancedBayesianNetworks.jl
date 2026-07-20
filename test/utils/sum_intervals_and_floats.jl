@testitem "Sum Intervals and Float function" begin
    a = Interval(0.2, 0.5)
    b = Interval(0.3, 0.4)
    c = 0.2
    @test EnhancedBayesianNetworks._sum_interval_and_floats(a, b, c) == (0.7, 1.1)
end