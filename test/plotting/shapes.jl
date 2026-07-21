@testitem "Gplot - Node shape" setup = [ExtraDeps, SetupPlotNet] begin
    @test _node_shape(W) == :rectangle
    @test _node_shape(S) == :rectangle
    @test _node_shape(U) == :circle
    @test _node_shape(D) == :circle
    @test _node_shape(R1) == :rounded_hexagon
    @test _node_shape(R2) == :rounded_hexagon
    @test _node_shape(E) == :hexagon
end

@testitem "Gplot - Hexagon vertices" setup = [ExtraDeps, SetupPlotNet] begin
    v = _hexagon_vertices(0.5, 0.4, 0.05, 0.0475)
    @test length(v) == 6
    @test all(extrema(first.(v)) .≈ (0.45, 0.55))      # spans the full 2hw box
    @test all(extrema(last.(v)) .≈ (0.3525, 0.4475))   # spans the full 2hh box
    @test sum(first, v) / 6 ≈ 0.5                      # centred
    @test sum(last, v) / 6 ≈ 0.4
    @test count(p -> p[2] ≈ 0.4475, v) == 2            # flat top: two vertices share it
    @test count(p -> p[2] ≈ 0.3525, v) == 2            # flat bottom
end

@testitem "Gplot - Rounded polygon" setup = [ExtraDeps, SetupPlotNet] begin
    v = _hexagon_vertices(0.5, 0.4, 0.05, 0.0475)
    r = _HEX_CORNER_RADIUS * 0.05
    p = _rounded_polygon(v, r, 8)
    @test length(p) == 6 * 9                                   # one 9-sample arc per corner
    @test all(0.45 - 1e-9 .<= first.(p) .<= 0.55 + 1e-9)       # never leaves the sharp box
    @test all(0.3525 - 1e-9 .<= last.(p) .<= 0.4475 + 1e-9)
    @test all(border_distance(q, v) <= r + 1e-9 for q in p)    # never deviates by more than r
    @test all(inside_polygon(q, v) for q in p)                 # rounding only cuts, never adds
    @test sum(first, p) / length(p) ≈ 0.5 atol = 1e-9          # stays centred
    @test sum(last, p) / length(p) ≈ 0.4 atol = 1e-9
    @test length(unique(_rounded_polygon(v, 0.0, 2))) == 6     # zero radius: back to the vertices
    @test length(_rounded_polygon(v, r, 3)) == 6 * 4           # arc sampling is configurable
    @test all(inside_polygon(q, v) for q in _rounded_polygon(v, 1.0, 4))  # oversized radius clamps
end

@testitem "Gplot - Circle and rectangle borders" setup = [ExtraDeps, SetupPlotNet] begin
    @test all(_circle_border(0.5, 0.4, 0.0, 0.05) .≈ (0.55, 0.4))
    @test all(_circle_border(0.5, 0.4, π, 0.05) .≈ (0.45, 0.4))
    for θ in range(-π, π; length=17)
        p = _circle_border(0.5, 0.4, θ, 0.05)
        @test hypot(p[1] - 0.5, p[2] - 0.4) ≈ 0.05             # always on the circumference
    end
    @test all(_rect_border(0.5, 0.4, 0.0, 0.05, 0.0475) .≈ (0.55, 0.4))
    @test all(_rect_border(0.5, 0.4, π / 2, 0.05, 0.0475) .≈ (0.5, 0.4475))
    @test all(_rect_border(0.5, 0.4, -π / 2, 0.05, 0.0475) .≈ (0.5, 0.3525))
    @test all(_rect_border(0.5, 0.4, π / 4, 0.05, 0.0475) .≈ (0.5475, 0.4475))  # corner
end

@testitem "Gplot - Polygon and hexagon borders" setup = [ExtraDeps, SetupPlotNet] begin
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    @test all(_polygon_border(0.5, 0.5, 0.0, square) .≈ (1.0, 0.5))
    @test all(_polygon_border(0.5, 0.5, π / 2, square) .≈ (0.5, 1.0))
    @test _polygon_border(5.0, 5.0, 0.0, square) == (5.0, 5.0)   # no hit: falls back to the centre

    # left/right vertices at ±hw, flat top/bottom edges at ±hh
    @test all(_hexagon_border(0.5, 0.4, 0.0, 0.05, 0.0475) .≈ (0.55, 0.4))
    @test all(_hexagon_border(0.5, 0.4, π / 2, 0.05, 0.0475) .≈ (0.5, 0.4475))
    @test all(_hexagon_border(0.5, 0.4, π, 0.05, 0.0475) .≈ (0.45, 0.4))
    @test all(_hexagon_border(0.5, 0.4, -π / 2, 0.05, 0.0475) .≈ (0.5, 0.3525))

    # for every direction the hit lies on the border and on the requested ray
    v = _hexagon_vertices(0.5, 0.4, 0.05, 0.0475)
    for θ in range(-π, π; length=73)
        p = _hexagon_border(0.5, 0.4, θ, 0.05, 0.0475)
        L = hypot(p[1] - 0.5, p[2] - 0.4)
        @test (p[1] - 0.5) / L ≈ cos(θ) atol = 1e-9
        @test (p[2] - 0.4) / L ≈ sin(θ) atol = 1e-9
        @test border_distance(p, v) < 1e-12
    end
end

@testitem "Gplot - Border point" setup = [ExtraDeps, SetupPlotNet] begin
    @test _border_point(D, 0.5, 0.4, 0.7, 0.05, 0.0475) == _circle_border(0.5, 0.4, 0.7, 0.05)
    @test _border_point(U, 0.5, 0.4, 0.7, 0.05, 0.0475) == _circle_border(0.5, 0.4, 0.7, 0.05)
    @test _border_point(W, 0.5, 0.4, 0.7, 0.05, 0.0475) == _rect_border(0.5, 0.4, 0.7, 0.05, 0.0475)
    @test _border_point(S, 0.5, 0.4, 0.7, 0.05, 0.0475) == _rect_border(0.5, 0.4, 0.7, 0.05, 0.0475)
    @test _border_point(E, 0.5, 0.4, 0.7, 0.05, 0.0475) == _hexagon_border(0.5, 0.4, 0.7, 0.05, 0.0475)
    @test _border_point(R1, 0.5, 0.4, 0.7, 0.05, 0.0475) == _hexagon_border(0.5, 0.4, 0.7, 0.05, 0.0475)
end

@testitem "Gplot - Node color" setup = [ExtraDeps, SetupPlotNet] begin
    @test _node_color(W) == "lightgreen"
    @test _node_color(D) == "lightgreen"
    @test _node_color(U) == "lightgreen"
    @test _node_color(S) == "limegreen"    # imprecise
    @test _node_color(R1) == "orange"
    @test _node_color(R2) == "orange"
    @test _node_color(E) == "orange"
end

@testitem "Gplot - Node stroke width" setup = [ExtraDeps, SetupPlotNet] begin
    @test _node_strokewidth(U) == 1.2mm    # discretized continuous node
    @test _node_strokewidth(D) == 0.3mm
    @test _node_strokewidth(W) == 0.3mm
    @test _node_strokewidth(S) == 0.3mm
    @test _node_strokewidth(E) == 0.3mm
    @test _node_strokewidth(R1) == 0.3mm   # empty ApproximatedDiscretization
end

@testitem "Gplot - Node form" setup = [ExtraDeps, SetupPlotNet] begin
    @test _node_form(D, 0.5, 0.4, 0.05, 0.0475) isa Compose.Form
    @test typeof(_node_form(D, 0.5, 0.4, 0.05, 0.0475)) == typeof(Compose.circle(0.5, 0.4, 0.05))
    @test typeof(_node_form(W, 0.5, 0.4, 0.05, 0.0475)) == typeof(Compose.rectangle(0.0, 0.0, 1.0, 1.0))
    @test npoints(_node_form(E, 0.5, 0.4, 0.05, 0.0475)) == 6      # pointy corners
    @test npoints(_node_form(R1, 0.5, 0.4, 0.05, 0.0475)) == 54    # rounded corners
end

@testitem "Gplot - Node contexts" setup = [ExtraDeps, SetupPlotNet] begin
    n = length(plotnet.nodes)
    ctxs = _build_node_contexts(fill(0.5, n), fill(0.5, n), plotnet.nodes, 0.05, 0.0475)
    @test ctxs isa Vector{Compose.Context}
    @test length(ctxs) == n
    @test isempty(_build_node_contexts(Float64[], Float64[], AbstractNode[], 0.05, 0.0475))
end