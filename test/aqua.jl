@testitem "Aqua" begin
    using Aqua
    # `undefined_exports` is disabled because UncertaintyQuantification exports names it does not define, and `@reexport` forwards them. `Context` is also  exported twice, by UncertaintyQuantification (from DifferentiationInterface) and by Compose, so it does not resolve here either.
    Aqua.test_all(EnhancedBayesianNetworks; undefined_exports = false)
end
