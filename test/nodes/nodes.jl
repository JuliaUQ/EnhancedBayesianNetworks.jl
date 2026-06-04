@testitem "Exact Discretization" begin
    interval = [-1, 0, 3, 1]
    @test_throws ErrorException("Invalid ExactDiscretization: interval values [-1, 0, 3, 1] are not sorted") ExactDiscretization(interval)
    interval = [-1, 0, 1, 3]
    exact_interval = ExactDiscretization([-1, 0, 1, 3])
    @test exact_interval.intervals == interval
    @test !isempty(exact_interval)
    @test isempty(ExactDiscretization())
end

@testitem "Approximated Discretization" begin
    interval = [-1, 0, 3, 1]
    sigma = 2
    @test_throws ErrorException("Invalid ApproximatedDiscretization: interval values [-1, 0, 3, 1] are not sorted") ApproximatedDiscretization(interval, sigma)
    interval = [-1, 0, 1, 3]
    sigma = -1
    @test_throws ErrorException("Invalid ApproximatedDiscretization: variance must be positive") ApproximatedDiscretization(interval, sigma)
    sigma = 10
    @test_logs (:warn, "Selected variance values $sigma could be too large for a realistic tails approximation") ApproximatedDiscretization(interval, sigma)
    sigma = 2
    approx_interval = ApproximatedDiscretization([-1, 0, 1, 3], 2)
    @test approx_interval.intervals == interval
    @test !isempty(approx_interval)
    @test isempty(ApproximatedDiscretization())
end
