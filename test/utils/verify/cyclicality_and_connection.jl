@testitem "Transitive Closure" setup=[ExtraDeps] begin
    rowsA = [1, 1, 2, 3, 4, 4]
    colsA = [3, 4, 6, 5, 5, 6]
    valsA = trues(6)
    A = sparse(rowsA, colsA, valsA, 6, 6)

    rowsR = [1, 1, 1, 1, 2, 3, 4, 4]
    colsR = [3, 4, 5, 6, 6, 5, 5, 6]
    valsR = trues(8)
    R = sparse(rowsR, colsR, valsR, 6, 6)

    @test EnhancedBayesianNetworks.transitive_closure(A) == R

    rowsA = [1, 2, 3, 4, 5]
    colsA = [3, 4, 5, 5, 1]
    valsA = trues(5)
    A = sparse(rowsA, colsA, valsA, 5, 5)

    rowsR = [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    colsR = [1, 3, 5, 1, 3, 4, 5, 1, 3, 5, 1, 3, 5, 1, 3, 5]
    valsR = trues(16)
    R = sparse(rowsR, colsR, valsR, 5, 5)

    @test EnhancedBayesianNetworks.transitive_closure(A) == R
end

@testitem "Iscyclic function" setup=[ExtraDeps] begin
    rowsA = [1, 2, 3, 4, 5]
    colsA = [3, 4, 5, 5, 1]
    valsA = trues(5)
    A = sparse(rowsA, colsA, valsA, 5, 5)
    @test EnhancedBayesianNetworks.iscyclic(A)

    rowsA = [1, 2, 3, 4]
    colsA = [3, 4, 5, 5]
    valsA = trues(4)
    A = sparse(rowsA, colsA, valsA, 5, 5)
    @test !EnhancedBayesianNetworks.iscyclic(A)
end

@testitem "Isconnected function" setup=[ExtraDeps] begin
    rowsA = [1, 2, 3, 4, 5]
    colsA = [3, 4, 5, 5, 1]
    valsA = trues(5)
    A = sparse(rowsA, colsA, valsA, 5, 5)
    @test EnhancedBayesianNetworks.isconnected(A)

    rowsA = [1, 2, 4]
    colsA = [3, 4, 5]
    valsA = trues(3)
    A = sparse(rowsA, colsA, valsA, 5, 5)
    @test !EnhancedBayesianNetworks.isconnected(A)
end