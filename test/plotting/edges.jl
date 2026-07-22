@testitem "Gplot - Arrow coordinates" setup = [ExtraDeps, SetupPlotNet] begin
    a1, a2 = _arrowcoords(0.0, 0.5, 0.4, 0.03, π / 9)
    @test _midpoint(a1, a2)[2] ≈ 0.4              # base centred on the arrow axis
    @test _midpoint(a1, a2)[1] < 0.5              # base sits behind the tip
    @test hypot(a1[1] - 0.5, a1[2] - 0.4) ≈ 0.03  # both barbs one arrowlength from the tip
    @test hypot(a2[1] - 0.5, a2[2] - 0.4) ≈ 0.03
    @test (a1[2] - 0.4) * (a2[2] - 0.4) < 0       # one barb each side of the axis
end

@testitem "Gplot - Midpoint" setup = [ExtraDeps, SetupPlotNet] begin
    @test _midpoint((0.0, 0.0), (1.0, 2.0)) == (0.5, 1.0)
    @test _midpoint((-1.0, -1.0), (1.0, 1.0)) == (0.0, 0.0)
    @test _midpoint((0.3, 0.7), (0.3, 0.7)) == (0.3, 0.7)
end

@testitem "Gplot - Build edges" setup = [ExtraDeps, SetupPlotNet] begin
    lines, arrows = _build_edges(
        Tuple{Int,Int}[],
        Float64[],
        Float64[],
        AbstractNode[],
        0.05,
        0.0475,
        0.03,
        π / 9
    )
    @test isempty(lines) && isempty(arrows)

    # W (rectangle) above E (hexagon), edge pointing straight down
    lines, arrows = _build_edges(
        [(1, 2)],
        [0.5, 0.5],
        [0.2, 0.8],
        [W, E],
        0.05,
        0.0475,
        0.03,
        π / 9
    )
    @test length(lines) == length(arrows) == 1
    @test all(lines[1][1] .≈ (0.5, 0.2475))                          # leaves W's bottom edge
    @test all(arrows[1][2] .≈ (0.5, 0.8 - 0.0475))                   # tip on E's top edge
    @test all(lines[1][2] .≈ _midpoint(arrows[1][1], arrows[1][3]))  # stops at the arrowhead base
    @test length(arrows[1]) == 3

    # D (circle) left of R1 (rounded hexagon), edge pointing right
    lines, arrows = _build_edges(
        [(1, 2)],
        [0.2, 0.8],
        [0.5, 0.5],
        [D, R1],
        0.05,
        0.0475,
        0.03,
        π / 9
    )
    @test all(lines[1][1] .≈ (0.25, 0.5))    # leaves D's circumference
    @test all(arrows[1][2] .≈ (0.75, 0.5))   # tip on R1's left vertex

    lines, arrows = _build_edges(
        [(1, 2), (1, 3)],
        [0.5, 0.2, 0.8],
        [0.2, 0.8, 0.8],
        [W, E, R1],
        0.05,
        0.0475,
        0.03,
        π / 9
    )
    @test length(lines) == length(arrows) == 2
end