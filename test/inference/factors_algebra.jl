@testitem "Factors Algebra - restrict" begin
    # test restrict - scalar
    f = Factor([1], [0.5, 0.5])
    r = EnhancedBayesianNetworks.restrict(f, 1, 1)
    @test r.vars == Int[]
    @test r.table[] == 0.5
    # test restrict - not scalar
    f = Factor([1, 2], [0.8 0.2; 0.1 0.9])
    r = EnhancedBayesianNetworks.restrict(f, 2, 1)
    @test r.vars == [1]
    @test size(r.table) == (2,)
    @test r.table[1] == 0.8
    @test r.table[2] == 0.1
end

@testitem "Factors Algebra - sumout" begin
    f = Factor([1], [0.5, 0.5])
    s = EnhancedBayesianNetworks.sumout(f, 1)
    @test s.vars == Int[]
    @test s.table[] == 1.0

    f = Factor([1, 2], [0.8 0.2; 0.1 0.9])
    s = EnhancedBayesianNetworks.sumout(f, 2)
    @test s.vars == [1]
    @test s.table[1] == 1.0
    @test s.table[2] == 1.0
end

@testitem "Factors Algebra - expand" begin
    # expand test - no expansion
    f = Factor([1, 2], reshape(collect(1:4), 2, 2))
    allpos = Dict(v => i for (i, v) in enumerate([1, 2]))
    A = EnhancedBayesianNetworks.expand(f, [1, 2], allpos)
    @test size(A) == (2, 2)
    @test collect(A) == f.table

    # expand test - Add missing variable
    f = Factor([2, 4], reshape(collect(1:4), 2, 2))
    allpos = Dict(v => i for (i, v) in enumerate([1, 2, 4]))
    A = EnhancedBayesianNetworks.expand(f, [1, 2, 4], allpos)
    @test size(A) == (1, 2, 2)
    @test A[1, 1, 1] == f.table[1, 1]
    @test A[1, 2, 1] == f.table[2, 1]
    @test A[1, 1, 2] == f.table[1, 2]
    @test A[1, 2, 2] == f.table[2, 2]

    # expand test - variable order
    f = Factor([4, 2], reshape(collect(1:4), 2, 2))
    allpos = Dict(v => i for (i, v) in enumerate([1, 2, 4]))
    A = EnhancedBayesianNetworks.expand(f, [1, 2, 4], allpos)
    expected = Array{Int}(undef, 1, 2, 2)
    expected[1, 1, 1] = 1
    expected[1, 1, 2] = 2
    expected[1, 2, 1] = 3
    expected[1, 2, 2] = 4
    @test all(collect(A) .== expected)
end

@testitem "Factors Algebra - reorder" begin
    f = Factor([1, 2], reshape(collect(1:4), 2, 2))
    r = EnhancedBayesianNetworks.reorder(f, [1, 2])

    @test r.vars == [1, 2]
    @test r.table == f.table

    f = Factor([1, 2], reshape(collect(1:4), 2, 2))
    r = EnhancedBayesianNetworks.reorder(f, [2, 1])
    @test r.vars == [2, 1]
    @test r.table[1, 1] == f.table[1, 1]
    @test r.table[2, 1] == f.table[1, 2]
    @test r.table[1, 2] == f.table[2, 1]
    @test r.table[2, 2] == f.table[2, 2]

    f = Factor([1, 2, 3], reshape(collect(1:24), 2, 3, 4))
    r = EnhancedBayesianNetworks.reorder(f, [3, 1, 2])
    @test r.vars == [3, 1, 2]
    @test size(r.table) == (4, 2, 3)
    @test r.table[1, 1, 1] == f.table[1, 1, 1]
    @test r.table[4, 2, 3] == f.table[2, 3, 4]

    f = Factor([1, 2, 3], reshape(collect(1:24), 2, 3, 4))
    r1 = EnhancedBayesianNetworks.reorder(f, [3, 1, 2])
    r2 = EnhancedBayesianNetworks.reorder(r1, [1, 2, 3])
    @test r2.vars == f.vars
    @test r2.table == f.table

    f = Factor([5], [0.3, 0.7])
    r = EnhancedBayesianNetworks.reorder(f, [5])
    @test r.vars == [5]
    @test r.table == [0.3, 0.7]

    f = Factor(Int[], fill(42.0))
    r = EnhancedBayesianNetworks.reorder(f, Int[])
    @test isempty(r.vars)
    @test r.table[] == 42.0

    f = Factor([10, 20, 30], reshape(collect(1:24), 2, 3, 4))
    r = EnhancedBayesianNetworks.reorder(f, [30, 10, 20])
    @test r.vars == [30, 10, 20]
    for i in 1:2
        for j in 1:3
            for k in 1:4
                @test f.table[i, j, k] == r.table[k, i, j]
            end
        end
    end
end

@testitem "Factors Algebra - multiply" begin
    fW = Factor([1], [0.5, 0.5])
    fR = Factor([1, 2], [0.8 0.2; 0.1 0.9])
    fWR = EnhancedBayesianNetworks.multiply(fW, fR)
    @test fWR.vars == [1, 2]
    @test size(fWR.table) == (2, 2)
    @test fWR.table[1, 1] == 0.4
    @test fWR.table[1, 2] == 0.1
    @test fWR.table[2, 1] == 0.05
    @test fWR.table[2, 2] == 0.45

    fS = Factor([1, 3], [0.4 0.4 0.2; 0.6000000000000001 0.30000000000000004 0.10000000000000002])
    f = EnhancedBayesianNetworks.multiply(fS, fR)
    @test f.vars == [1, 3, 2]
    @test size(f.table) == (2, 3, 2)

    # commutativity
    A = EnhancedBayesianNetworks.multiply(fW, fR)
    B = EnhancedBayesianNetworks.multiply(fR, fW)
    @test sort(A.vars) == sort(B.vars)

    # constant factor
    c = Factor(Int[], fill(0.5))
    f = Factor([1], [0.5, 0.5])
    m = EnhancedBayesianNetworks.multiply(c, f)
    @test m.vars == [1]
    @test m.table == [0.25, 0.25]

    # multiply all - 2 factors
    fWR = EnhancedBayesianNetworks.multiply([fW, fR])
    @test fWR.vars == [1, 2]
    @test size(fWR.table) == (2, 2)
    @test fWR.table[1, 1] == 0.4
    @test fWR.table[1, 2] == 0.1
    @test fWR.table[2, 1] == 0.05
    @test fWR.table[2, 2] == 0.45

    a = EnhancedBayesianNetworks.multiply([fW, fR])
    b = EnhancedBayesianNetworks.multiply(fW, fR)
    @test a.table == b.table
    @test a.vars == b.vars

    # multiply all - 3 factors
    f = EnhancedBayesianNetworks.multiply([fW, fR, fS])
    f.vars == [1, 2, 3]
    @test size(f.table) == (2, 2, 3)
    @test f.table[1, 1, 1] ≈ 0.16
    @test f.table[2, 2, 3] ≈ 0.045

    # multiply all - 1 factor
    m = EnhancedBayesianNetworks.multiply([fW])
    @test m.vars == fW.vars
    @test m.table == fW.table

    # error message
    @test_throws ErrorException("Cannot multiply an empty factor set") EnhancedBayesianNetworks.multiply(Factor[])
end

@testitem "Factors Algebra - normalize" begin
    # normalize - 1D
    f = Factor([1], [2.0, 3.0])
    n = EnhancedBayesianNetworks.normalize(f)
    @test n.vars == [1]
    @test n.table ≈ [0.4, 0.6]
    @test sum(n.table) ≈ 1.0

    # normalize - 2D
    f = Factor([1, 2], [1.0 2.0; 3.0 4.0])
    n = EnhancedBayesianNetworks.normalize(f)
    @test sum(n.table) ≈ 1.0
    @test n.table ≈ [0.1 0.2; 0.3 0.4]

    # normalize - constant factor
    f = Factor(Int[], fill(5.0))
    n = EnhancedBayesianNetworks.normalize(f)
    @test n.vars == Int[]
    @test n.table[] ≈ 1.0
end