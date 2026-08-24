## Description of file structure

* __nodes__: the node types — discrete, continuous, and functional — and their conditional probability tables (`ScenariosTable`)
* __networks__: the three network types — `bn` (Bayesian Network), `cn` (Credal Network), and `ebn` (Enhanced Bayesian Network, with the `reduction` pipeline) — plus shared construction/validation (`networks_common.jl`) and type dispatch (`dispatch.jl`)
* __inference__: exact inference by variable elimination (factors and their algebra, elimination ordering, and `infer`)
* __learning__: parameter learning from data — maximum likelihood, Dirichlet/Laplace smoothing, and Expectation-Maximization
* __show__: `Base.show` methods for nodes, networks, and posteriors
* __plotting__: network visualization — node shapes, edges, labels, layout, and legend
* __utils__: miscellaneous helper functions — topological sort, cyclicality/connection checks, interval/float utilities

__EnhancedBayesianNetworks.jl__ is the main Julia module file
