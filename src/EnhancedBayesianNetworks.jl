module EnhancedBayesianNetworks

using DataFrames
using Distributions
using LinearAlgebra
using Polyhedra: HalfSpace, doubledescription
using Reexport
using SparseArrays
using Compose

@reexport using UncertaintyQuantification

import Base: *, sum, reduce

# Types
export ApproximatedDiscretization
export BayesianNetwork
export ContinuousFunctionalNode
export ContinuousNode
export CredalNetwork
export CredalPosterior
export DirectAcyclicGraph
export DiscreteFunctionalNode
export DiscreteNode
export ExactDiscretization
export EnhancedBayesianNetwork
export Evidence
export Posterior

# Constants
const Evidence = Dict{Symbol,Symbol}

# Functions
export add_child!
export add_node!
export children
export discrete_ancestors
export factor_score
export fill_factor_score
export fill_score
export gplot
export infer
export isprecise
export isroot
export joint_probability
export markov_blanket
export markov_envelope
export order!
export parents
export reduce
export sample
export saveplot
export scenarios
export states

include("nodes/nodes.jl")
include("networks/networks.jl")
include("inference/inference.jl")
include("learning/learning.jl")
include("utils/base_show.jl")
include("utils/gplot.jl")
end