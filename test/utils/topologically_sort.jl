@testset "Topological Sorting" begin
    A = spzeros(Bool, 4, 4)
    A[1, 3] = true
    A[2, 3] = true
    A[3, 4] = true
    @test EnhancedBayesianNetworks.topologically_sort(A) == [1, 2, 3, 4]
    A = spzeros(Bool, 4, 4)
    A[2, 4] = true
    A[1, 4] = true
    A[4, 3] = true
    @test EnhancedBayesianNetworks.topologically_sort(A) == [1, 2, 4, 3]
end