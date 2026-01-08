@testset "Discretization structs" begin
    @testset "Exact Discretization" begin
        interval = [-1, 0, 3, 1]
        @test_throws ErrorException("interval values [-1, 0, 3, 1] are not sorted") ExactDiscretization(interval)
        interval = [-1, 0, 1, 3]
        exact_interval = ExactDiscretization([-1, 0, 1, 3])
        @test exact_interval.intervals == interval
    end
    @testset "Approximated Discretization" begin
        interval = [-1, 0, 3, 1]
        sigma = 2
        @test_throws ErrorException("interval values [-1, 0, 3, 1] are not sorted") ApproximatedDiscretization(interval, sigma)
        interval = [-1, 0, 1, 3]
        sigma = -1
        @test_throws ErrorException("variance must be positive") ApproximatedDiscretization(interval, sigma)
        sigma = 10
        @test_logs (:warn, "Selected variance values $sigma can be too big, and the approximation not realistic") ApproximatedDiscretization(interval, sigma)
        sigma = 2
        approx_interval = ApproximatedDiscretization([-1, 0, 1, 3], 2)
        @test approx_interval.intervals == interval
    end
end