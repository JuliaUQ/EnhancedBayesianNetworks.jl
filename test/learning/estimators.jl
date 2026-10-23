@testitem "Estimator constructors" setup = [ExtraDeps] begin
    @test PreciseDirichletModel().α == 0.0
    @test PreciseDirichletModel(1).α == 1.0
    @test ImpreciseDirichletModel().s == 1.0
    @test QuesenberryHurst().confidence == 0.95

    # every field is stored as Float64 whatever the argument, so the formulas never promote per call
    @test PreciseDirichletModel(0).α isa Float64
    @test ImpreciseDirichletModel(2).s isa Float64

    @test_throws ErrorException("Invalid PreciseDirichletModel: α must be finite and nonnegative, got -1") PreciseDirichletModel(-1)
    @test_throws ErrorException("Invalid PreciseDirichletModel: α must be finite and nonnegative, got Inf") PreciseDirichletModel(Inf)

    # s = 0 is excluded, not merely negative s: at s = 0 the interval collapses to a point and the estimator would silently stop being imprecise while still producing a CredalNetwork
    @test_throws ErrorException("Invalid ImpreciseDirichletModel: s must be finite and strictly positive, got 0") ImpreciseDirichletModel(0)
    @test_throws ErrorException("Invalid ImpreciseDirichletModel: s must be finite and strictly positive, got -1") ImpreciseDirichletModel(-1)

    @test_throws ErrorException("Invalid QuesenberryHurst: confidence must lie strictly between 0 and 1, got 0") QuesenberryHurst(0)
    @test_throws ErrorException("Invalid QuesenberryHurst: confidence must lie strictly between 0 and 1, got 1") QuesenberryHurst(1)

    # the type hierarchy is what `learn` dispatches the missing-data treatment on
    @test PreciseDirichletModel() isa PreciseEstimator
    @test ImpreciseDirichletModel() isa ImpreciseEstimator
    @test QuesenberryHurst() isa ImpreciseEstimator
end

@testitem "ConfigurationCounts constructor" setup = [ExtraDeps] begin
    # parametric over the element type: integer counts off a data frame, fractional ones from the responsibilities an EM step produces, and `_estimate` must take both
    complete = EnhancedBayesianNetworks.ConfigurationCounts([3, 1], 4, 0, [0, 0], 0)
    @test complete isa EnhancedBayesianNetworks.ConfigurationCounts{Int}
    @test complete.counts == [3, 1]
    @test complete.total == 4

    weighted = EnhancedBayesianNetworks.ConfigurationCounts([1.5, 2.5], 4.0, 0.0, [0.0, 0.0], 0.0)
    @test weighted isa EnhancedBayesianNetworks.ConfigurationCounts{Float64}
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(1), weighted) ≈ [(1.5 + 1) / (4 + 2), (2.5 + 1) / (4 + 2)]
end

@testitem "Estimators on complete data" setup = [ExtraDeps] begin
    counts(c) = EnhancedBayesianNetworks.ConfigurationCounts(c, sum(c), 0, zeros(Int, length(c)), 0)

    # α = 0 is the relative frequency and gives an unobserved state exactly 0
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(0), counts([3, 1])) == [0.75, 0.25]
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(0), counts([4, 0])) == [1.0, 0.0]
    # α > 0 pulls towards the uniform: (cnt + α) / (total + α * k)
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(1), counts([3, 1])) ≈ [4 / 6, 2 / 6]

    @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), counts([3, 1])) == [Interval(3 / 5, 4 / 5), Interval(1 / 5, 2 / 5)]
    # the IDM interval always contains PreciseDirichletModel(s / k), the uniform-prior case
    point = EnhancedBayesianNetworks._estimate(PreciseDirichletModel(1 / 2), counts([3, 1]))
    interval = EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), counts([3, 1]))
    @test all(i.lb <= p <= i.ub for (p, i) in zip(point, interval))

    # coherence: the credal set is never empty, which is what `_verify_exhaustiveness` demands
    for c in ([3, 1], [5, 0], [2, 2, 2], [7, 1, 0, 0])
        iv = EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), counts(c))
        @test sum(i.lb for i in iv) <= 1 <= sum(i.ub for i in iv)
    end

    # a node with a single state is certain under every estimator, and returns a number rather than a degenerate interval. A DAG of such nodes therefore stays a BayesianNetwork however it was fitted
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(), counts([7])) == [1.0]
    @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(), counts([7])) == [1.0]
    @test EnhancedBayesianNetworks._estimate(QuesenberryHurst(), counts([7])) == [1.0]

    # a parent configuration absent from the data: uniform if precise, vacuous if imprecise
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(0), counts([0, 0, 0])) ≈ [1 / 3, 1 / 3, 1 / 3]
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(1), counts([0, 0, 0])) ≈ [1 / 3, 1 / 3, 1 / 3]
    @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), counts([0, 0])) == [Interval(0.0, 1.0), Interval(0.0, 1.0)]
    @test EnhancedBayesianNetworks._estimate(QuesenberryHurst(), counts([0, 0])) == [Interval(0.0, 1.0), Interval(0.0, 1.0)]
end

@testitem "QuesenberryHurst against the Wilson score interval" setup = [ExtraDeps] begin
    counts(c) = EnhancedBayesianNetworks.ConfigurationCounts(c, sum(c), 0, zeros(Int, length(c)), 0)

    # at k = 2 the chi-square quantile is z^2, so the projected ellipsoid is exactly the Wilson score interval
    function wilson(cnt, n, confidence)
        z = quantile(Normal(), 1 - (1 - confidence) / 2)
        p = cnt / n
        centre = (p + z^2 / (2n)) / (1 + z^2 / n)
        halfwidth = z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / (1 + z^2 / n)
        return (centre - halfwidth, centre + halfwidth)
    end
    for (cnt, n) in [(3, 10), (1, 4), (7, 9), (50, 200), (2, 3)], confidence in (0.95, 0.99)
        got = EnhancedBayesianNetworks._estimate(QuesenberryHurst(confidence), counts([cnt, n - cnt]))[1]
        lower, upper = wilson(cnt, n, confidence)
        @test got.lb ≈ lower atol = 1.0e-12
        @test got.ub ≈ upper atol = 1.0e-12
    end

    # a state observed in every row must reach exactly 1. Without the pin in `_qh_interval` the addition `A + 2 * cnt + spread` rounds before `spread` is added.
    for n in (1, 4, 17, 100)
        @test EnhancedBayesianNetworks._estimate(QuesenberryHurst(), counts([n, 0]))[1].ub == 1.0
    end

    # a wider confidence level gives a wider interval
    narrow = EnhancedBayesianNetworks._estimate(QuesenberryHurst(0.9), counts([3, 7]))[1]
    wide = EnhancedBayesianNetworks._estimate(QuesenberryHurst(0.99), counts([3, 7]))[1]
    @test wide.lb < narrow.lb && narrow.ub < wide.ub
end

@testitem "ImpreciseDirichletModel on incomplete data" setup = [ExtraDeps] begin
    # three complete rows, one with G unknown, two whose parents are incomplete, two with neither observed
    frequencies = EnhancedBayesianNetworks.ConfigurationCounts([2, 1], 3, 1, [2, 1], 2)
    got = EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), frequencies)

    # n_bar = node_missing + parents_missing[k] + both_missing, n_low swaps in the other states' parents_missing, so the bounds are 2 / (3 + 4 + 1) and (2 + 5 + 1) / (3 + 5 + 1)
    @test got[1] == Interval(2 / 8, 8 / 9)
    @test got[2] == Interval(1 / 9, 6 / 8)

    # with nothing incomplete the virtual frequencies vanish and the estimator is the plain IDM
    for c in ([3, 1], [0, 0], [5, 2, 1])
        plain = EnhancedBayesianNetworks.ConfigurationCounts(c, sum(c), 0, zeros(Int, length(c)), 0)
        @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), plain) ==
            [Interval(x / (sum(c) + 1), (x + 1) / (sum(c) + 1)) for x in c]
    end

    # incompleteness can only widen: every extra ambiguous row moves the bounds apart
    base = EnhancedBayesianNetworks.ConfigurationCounts([2, 1], 3, 0, [0, 0], 0)
    more = EnhancedBayesianNetworks.ConfigurationCounts([2, 1], 3, 2, [0, 0], 0)
    b, m = EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), base), EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), more)
    @test all(m[i].lb <= b[i].lb && b[i].ub <= m[i].ub for i in 1:2)
    @test any(m[i].ub - m[i].lb > b[i].ub - b[i].lb for i in 1:2)
end

@testitem "Estimator guards against drifting counts" setup = [ExtraDeps] begin
    # The clamps and the `max` under the square root exist for one caller: an EM step, whose counts are sums of responsibilities and can exceed `total` by floating-point dust. The inputs below exaggerate that drift wildly, because a sub-ulp excess cannot be written down portably; they violate the `total == sum(counts)` invariant on purpose and are unreachable through `learn`.
    A = quantile(Chisq(1), 0.95)
    counts(c) = EnhancedBayesianNetworks.ConfigurationCounts(c, sum(c), 0, zeros(Int, length(c)), 0)
    @test EnhancedBayesianNetworks._qh_interval(10, 4, A) == Interval(1.0, 1.0)

    drifted = EnhancedBayesianNetworks.ConfigurationCounts([5.0, 0.0], 4.0, 0.0, [0.0, 0.0], 0.0)
    @test EnhancedBayesianNetworks._estimate(PreciseDirichletModel(0), drifted) == [1.0, 0.0]

    drifted = EnhancedBayesianNetworks.ConfigurationCounts([10.0, 0.0], 4.0, 0.0, [0.0, 0.0], 0.0)
    @test EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(1), drifted)[1] == Interval(1.0, 1.0)

    # on valid input an imprecise estimator can never return a degenerate interval: the IDM width is  s / (total + s) with s > 0 forbidden from being zero, and the QH spread is at least A > 0. This is what makes `isprecise` trustworthy, and with it the network type `_dispatch` chooses
    for c in ([3, 1], [0, 0], [40, 0], [24, 39, 39, 37, 36]), s in (0.5, 1.0, 2.0)
        @test all(i.ub > i.lb for i in EnhancedBayesianNetworks._estimate(ImpreciseDirichletModel(s), counts(c)))
    end
    for c in ([3, 1], [40, 0], [24, 39, 39, 37, 36]), conf in (0.9, 0.99)
        @test all(i.ub > i.lb for i in EnhancedBayesianNetworks._estimate(QuesenberryHurst(conf), counts(c)))
    end
end
