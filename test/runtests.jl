using Test
using TestItems
using TestItemRunner

using EnhancedBayesianNetworks
using Suppressor
using CSV

@run_package_tests()

include("utils/wrap.jl")
include("utils/flat.jl")
include("utils/topologically_sort.jl")
include("nodes/scenariostable.jl")
include("nodes/nodes.jl")
include("nodes/discretenodes.jl")
include("nodes/continuousnode.jl")
include("nodes/functionalnodes.jl")
include("utils/require_unique.jl")
include("utils/sum_intervals_and_float.jl")
include("utils/verify/add_child.jl")
include("utils/verify/cyclicality_and_connection.jl")
include("networks/networks_common.jl")
include("networks/bn/bayesnet.jl")
include("networks/cn/credalnet.jl")
include("networks/ebn/enhancedbn.jl")
include("networks/ebn/discretization/discretize.jl")
include("networks/ebn/reduction/evaluate_node.jl")
include("networks/ebn/reduction/reduce_net.jl")
include("networks/dispatch.jl")
include("inference/utils.jl")
include("inference/inference.jl")
include("inference/factors.jl")
include("inference/factors_algebra.jl")
include("inference/sorting.jl")
include("inference/variableselimination.jl")
include("inference/infer.jl")