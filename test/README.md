To run tests, from the top directory of EnhancedBayesianNetworks.jl either:

using the package manager:

```julia
shell> julia --project

julia> using Pkg
julia> Pkg.test()
```

or using the package REPL

```julia
shell> julia --project

julia>] test
```

or (one liner)

```
shell> julia --project -e 'using Pkg; Pkg.test()'
```

The test suite is organized with [TestItemRunner](https://github.com/julia-vscode/TestItemRunner.jl): each test is a self-contained `@testitem`, discovered and run by `@run_package_tests` in `runtests.jl`. Individual test items can also be run from the VS Code Test Explorer.
