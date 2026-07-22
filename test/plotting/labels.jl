@testitem "Gplot - Labels" setup = [ExtraDeps, SetupPlotNet] begin
    @test length(_build_labels([D], [0.5], [0.5], 8pt, 1.0)) == 1   # name only
    @test length(_build_labels([U], [0.5], [0.5], 8pt, 1.0)) == 1
    @test length(_build_labels([W], [0.5], [0.5], 8pt, 1.0)) == 2   # name + state count
    @test length(_build_labels([E], [0.5], [0.5], 8pt, 1.0)) == 2   # discrete functional has states
    @test isempty(_build_labels(AbstractNode[], Float64[], Float64[], 8pt, 1.0))

    n = length(plotnet.nodes)
    labels = _build_labels(plotnet.nodes, fill(0.5, n), fill(0.5, n), 8pt, 1.0)
    ndiscrete = count(x -> x isa AbstractDiscreteNode, plotnet.nodes)
    @test isa(labels, Vector{Compose.Context})
    @test length(labels) == 2 * ndiscrete + (n - ndiscrete)
end