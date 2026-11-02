@testitem "Exact Discretization" begin
    interval = [-1, 0, 3, 1]
    @test_throws ErrorException("Invalid ExactDiscretization: interval values [-1, 0, 3, 1] are not sorted") ExactDiscretization(interval)
    interval = [-1, 0, 1, 3]
    exact_interval = ExactDiscretization([-1, 0, 1, 3])
    @test exact_interval.intervals == interval
    @test !isempty(exact_interval)
    @test isempty(ExactDiscretization())

    # p-box mass settings: defaults, keyword forms (also on the empty strategy) and validation
    @test exact_interval.mesh == EnhancedBayesianNetworks.PBOX_MASS_MESH
    @test exact_interval.technique == :optimised
    @test ExactDiscretization(interval; mesh = 1.0e-4).mesh == 1.0e-4
    @test ExactDiscretization(interval; technique = :distribution_free).technique == :distribution_free
    @test ExactDiscretization(; technique = :distribution_free).technique == :distribution_free
    @test isempty(ExactDiscretization(; mesh = 1.0e-4))
    @test_throws ErrorException("Invalid discretization: p-box mass mesh must be positive, got 0.0") ExactDiscretization(interval; mesh = 0.0)
    @test_throws ErrorException("Invalid discretization: p-box mass mesh must be positive, got -1.0") ExactDiscretization(interval; mesh = -1.0)
    @test_throws ErrorException("Invalid discretization: unknown p-box mass technique :optimized, must be one of :optimised, :distribution_free") ExactDiscretization(interval; technique = :optimized)
end

@testitem "Approximated Discretization" begin
    interval = [-1, 0, 3, 1]
    scale = 2
    @test_throws ErrorException("Invalid ApproximatedDiscretization: interval values [-1, 0, 3, 1] are not sorted") ApproximatedDiscretization(interval, scale)
    interval = [-1, 0, 1, 3]
    scale = -1
    @test_throws ErrorException("Invalid ApproximatedDiscretization: scale parameter must be non-negative") ApproximatedDiscretization(interval, scale)
    scale = 10
    @test_logs (:warn, "Selected scale parameter $scale could be too large for a realistic tails approximation") ApproximatedDiscretization(interval, scale)
    scale = 2
    approx_interval = ApproximatedDiscretization([-1, 0, 1, 3], 2)
    @test approx_interval.intervals == interval
    @test !isempty(approx_interval)
    @test isempty(ApproximatedDiscretization())

    # p-box mass settings: defaults, keyword forms (also on the empty strategy) and validation
    @test approx_interval.mesh == EnhancedBayesianNetworks.PBOX_MASS_MESH
    @test approx_interval.technique == :optimised
    @test ApproximatedDiscretization(interval, scale; mesh = 1.0e-4).mesh == 1.0e-4
    @test ApproximatedDiscretization(interval, scale; technique = :distribution_free).technique == :distribution_free
    @test ApproximatedDiscretization(; technique = :distribution_free).technique == :distribution_free
    @test_throws ErrorException("Invalid discretization: p-box mass mesh must be positive, got 0.0") ApproximatedDiscretization(interval, scale; mesh = 0.0)
    @test_throws ErrorException("Invalid discretization: unknown p-box mass technique :free, must be one of :optimised, :distribution_free") ApproximatedDiscretization(interval, scale; technique = :free)
end
