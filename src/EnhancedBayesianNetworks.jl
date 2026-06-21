module EnhancedBayesianNetworks

using DataFrames
using Distributions
using LinearAlgebra
using NetworkLayout
using Polyhedra: HalfSpace, doubledescription
using Reexport
using SparseArrays
using UncertaintyQuantification

# @reexport using Graphs
@reexport using UncertaintyQuantification
@reexport using DataFrames
@reexport using SparseArrays
@reexport using Compose

import Base: *, sum, reduce

# Types
export ApproximatedDiscretization
export BayesianNetwork
export ContinuousFunctionalNode
export ContinuousInput
export ContinuousNode
export CredalNetwork
export CredalPosterior
export DiscreteFunctionalNode
export DiscreteNode
export ExactDiscretization
export EnhancedBayesianNetwork
export Evidence
export Factor
export FunctionalNode
export InferenceState
export Posterior

# export Factor

## Constants
const Evidence = Dict{Symbol,Symbol}

## Functions
export add_child!
export children
export discrete_ancestors
export factorize
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
export saveplot
export scenarios
export states

include("nodes/nodes.jl")
include("networks/networks.jl")
include("inference/inference.jl")
# include("learning/learning.jl")
include("utils/base_show.jl")
include("utils/plots.jl")
end