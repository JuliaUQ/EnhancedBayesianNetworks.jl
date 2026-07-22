@testitem "Gplot - Legend rows" setup = [ExtraDeps, SetupPlotNet] begin
    @test _legend_y(1) ≈ 0.05
    @test issorted(_legend_y.(1:10))
    @test all(0 .< _legend_y.(1:10) .< 1)          # all ten rows fit inside the legend box
    @test _legend_header(1, "Shape", 10) isa Compose.Context
    @test _legend_label(0.5, "Discrete", 9) isa Compose.Context
    @test _legend_shape(y -> Compose.circle(0.11, y, 0.035), 2, "Continuous", 9) isa Compose.Context
    @test _legend_shape(y -> Compose.circle(0.11, y, 0.035), 6, "Discretized", 9; fillcolor="lightgreen", linew=1.2mm) isa Compose.Context
    @test _legend_color(8, "Precise", 9, "lightgreen", 0.07, 0.014) isa Compose.Context
end

@testitem "Gplot - Legend" setup = [ExtraDeps, SetupPlotNet] begin
    @test _build_legend(1.0) isa Compose.Context
    @test _build_legend(0.5) isa Compose.Context
    @test _build_legend(1.0; x_fraction=0.1, y_fraction=0.1) isa Compose.Context
end