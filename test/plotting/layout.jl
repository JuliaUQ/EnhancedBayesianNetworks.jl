@testitem "Gplot - Compute layers" setup = [ExtraDeps, SetupPlotNet] begin
    @test _compute_layers(spzeros(Bool, 3, 3)) == [0, 0, 0]      # all roots

    chain = spzeros(Bool, 3, 3)
    chain[1, 2] = chain[2, 3] = true
    @test _compute_layers(chain) == [0, 1, 2]

    diamond = spzeros(Bool, 4, 4)
    diamond[1, 2] = diamond[1, 3] = diamond[2, 4] = diamond[3, 4] = true
    @test _compute_layers(diamond) == [0, 1, 1, 2]

    # a node sits one row below its DEEPEST parent, not its shallowest
    skip = spzeros(Bool, 4, 4)
    skip[1, 2] = skip[2, 3] = skip[1, 4] = skip[3, 4] = true
    @test _compute_layers(skip) == [0, 1, 2, 3]
end

@testitem "Gplot - Layered positions" setup = [ExtraDeps, SetupPlotNet] begin
    diamond = spzeros(Bool, 4, 4)
    diamond[1, 2] = diamond[1, 3] = diamond[2, 4] = diamond[3, 4] = true

    x, y = _layered_positions(diamond, _BORDER_PAD, _BORDER_PAD)
    @test x[1] ≈ 0.5 && x[4] ≈ 0.5                       # alone on their row: centred
    @test x[2] ≈ _BORDER_PAD && x[3] ≈ 1 - _BORDER_PAD   # sharing a row: spread to the edges
    @test y[1] < y[2] == y[3] < y[4]                     # one row per layer
    @test all(_BORDER_PAD - 1e-9 .<= x .<= 1 - _BORDER_PAD + 1e-9)
    @test all(_BORDER_PAD - 1e-9 .<= y .<= 1 - _BORDER_PAD + 1e-9)

    # top_pad pushes the first row down and squeezes the rest
    x2, y2 = _layered_positions(diamond, _BORDER_PAD, 0.18)
    @test x2 == x
    @test y2[1] ≈ 0.18
    @test y2[4] ≈ 1 - _BORDER_PAD

    xs, ys = _layered_positions(spzeros(Bool, 1, 1), _BORDER_PAD, _BORDER_PAD)
    @test xs ≈ [0.5] && ys ≈ [_BORDER_PAD]
end