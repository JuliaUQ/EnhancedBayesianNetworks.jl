@testsnippet SetupPlotNet begin
    using EnhancedBayesianNetworks: Compose, cm, mm, pt,
        _node_shape, _hexagon_vertices, _rounded_polygon,
        _circle_border, _rect_border, _polygon_border, _hexagon_border, _border_point,
        _node_color, _node_strokewidth, _node_form, _build_node_contexts,
        _compute_layers, _layered_positions, _arrowcoords, _midpoint, _build_edges,
        _build_labels, _legend_y, _legend_label, _legend_header, _legend_shape,
        _legend_color, _build_legend, _BORDER_PAD, _HEX_CORNER_RADIUS,
        AbstractNode, AbstractDiscreteNode

    # A net holding every shape: discrete, imprecise discrete, discretized
    # continuous, plain continuous, and both flavours of functional node.
    W = DiscreteNode(:W)
    W[:W=>:sunny] = 0.7
    W[:W=>:rainy] = 0.3

    S = DiscreteNode(:S, [:W])
    S[:W=>:sunny, :S=>:on] = Interval(0.1, 0.3)
    S[:W=>:sunny, :S=>:off] = Interval(0.7, 0.9)
    S[:W=>:rainy, :S=>:on] = Interval(0.4, 0.6)
    S[:W=>:rainy, :S=>:off] = Interval(0.4, 0.6)

    U = ContinuousNode(:U, [:W], ApproximatedDiscretization([-1.0, 1.0], 2))
    U[:W=>:sunny] = Normal()
    U[:W=>:rainy] = Normal(2, 1)

    D = ContinuousNode(:D)
    D[] = Normal(1, 2)

    R1 = ContinuousFunctionalNode(:R1, [Model(df -> df.U .^ 2, :r1)], MonteCarlo(100))
    R2 = ContinuousFunctionalNode(:R2, [Model(df -> df.D .+ 1, :r2)], MonteCarlo(100))
    E = DiscreteFunctionalNode(:E, [Model(df -> df.r1 .+ df.r2, :G)], df -> df.G .- 1.0, MonteCarlo(100))

    plotnet = EnhancedBayesianNetwork([W, S, U, D, R1, R2, E])
    add_child!(plotnet, W, S)
    add_child!(plotnet, W, U)
    add_child!(plotnet, U, R1)
    add_child!(plotnet, D, R2)
    add_child!(plotnet, [R1, R2], E)
    order!(plotnet)

    # Distance from a point to a segment, and to the nearest edge of a polygon:
    # used to assert that a computed attachment point really sits on the border.
    function segment_distance(p, a, b)
        ex, ey = b[1] - a[1], b[2] - a[2]
        L2 = ex^2 + ey^2
        t = L2 < 1e-24 ? 0.0 : clamp(((p[1] - a[1]) * ex + (p[2] - a[2]) * ey) / L2, 0.0, 1.0)
        return hypot(p[1] - (a[1] + t * ex), p[2] - (a[2] + t * ey))
    end

    border_distance(p, verts) = minimum(segment_distance(p, verts[k], verts[mod1(k + 1, length(verts))]) for k in eachindex(verts))

    # true when `p` is inside a convex polygon: every edge sees it on the same side
    function inside_polygon(p, verts)
        n = length(verts)
        crosses = [(verts[mod1(k + 1, n)][1] - verts[k][1]) * (p[2] - verts[k][2]) - (verts[mod1(k + 1, n)][2] - verts[k][2]) * (p[1] - verts[k][1]) for k in 1:n]
        return all(>=(-1e-12), crosses) || all(<=(1e-12), crosses)
    end

    npoints(form) = length(form.primitives[1].points)
end

@testitem "Gplot - Plot" setup = [ExtraDeps, SetupPlotNet] begin
    @test gplot(plotnet) isa Compose.Context
    @test gplot(plotnet; legend=true) isa Compose.Context
    @test gplot(plotnet; title="a title") isa Compose.Context
    @test gplot(plotnet; node_scale=1.5, label_scale=2.0, title_scale=0.5) isa Compose.Context
    @test gplot(plotnet; legend=true, legend_scale=0.8, legend_x=0.1, legend_y=0.1) isa Compose.Context
    @test gplot(plotnet; figsize=(10cm, 30cm)) isa Compose.Context

    dag = DirectAcyclicGraph()
    add_node!(dag, :A)
    add_node!(dag, :B; parents=[:A])
    @test gplot(dag) isa Compose.Context
end

@testitem "Gplot - Saveplot" setup = [ExtraDeps, SetupPlotNet] begin
    mktempdir() do dir
        f = joinpath(dir, "net.svg")
        saveplot(gplot(plotnet; legend=true, title="net"), f)
        @test isfile(f)
        @test filesize(f) > 0
        @test occursin("<svg", read(f, String))
    end
end