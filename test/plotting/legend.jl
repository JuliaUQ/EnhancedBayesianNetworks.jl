@testitem "Gplot - Legend" setup = [ExtraDeps, SetupPlotNet] begin
    pt_to_units = (2.54 / 72) / 20
    @test _build_legend(0.65, 0.6, 9, pt_to_units) isa Compose.Context
    @test _build_legend(0.1, 0.1, 6, pt_to_units) isa Compose.Context
    @test _build_legend(0.65, 0.6, 14, pt_to_units) isa Compose.Context
end
