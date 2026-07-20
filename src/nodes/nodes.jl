# Node type hierarchy: every node is discrete- or continuous-flavoured;
abstract type AbstractNode end
abstract type AbstractContinuousNode <: AbstractNode end
abstract type AbstractDiscreteNode <: AbstractNode end

# Discretization strategy attached to a continuous node (see ExactDiscretization / ApproximatedDiscretization).
abstract type AbstractDiscretization end

"""
    ExactDiscretization(intervals=Real[])

Discretization strategy for a continuous **root** node, allowing evidence to be observed on it.
The node's distribution support is partitioned exactly at the sorted `intervals` edges, turning the
continuous root into discrete bins. The default (empty `intervals`) leaves the node continuous, i.e.
no discretization is applied. The edges must be sorted.

# Examples
```julia
# discretize a root node's support at the edges -2, 0, 2:
disc = ExactDiscretization([-2.0, 0.0, 2.0])
T = ContinuousNode(:T, Normal(), disc)

ExactDiscretization()            # empty: the root stays continuous
```
"""
struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}

    function ExactDiscretization(intervals::Vector{<:Real})
        if !issorted(intervals)
            error("Invalid ExactDiscretization: interval values $intervals are not sorted")
        end
        new(intervals)
    end
end

ExactDiscretization() = ExactDiscretization(Vector{Real}())

"""
    ApproximatedDiscretization(intervals=Real[], sigma=0)

Discretization strategy for a continuous **non-root (child)** node, allowing evidence to be observed
on it. The sorted `intervals` edges partition the support into discrete bins, while `sigma` is the
spread of the normal distribution used to approximate the original continuous distribution's tails
when a discrete state is mapped back to a continuous range. `sigma` must be non-negative; a value
above `2` is accepted but warns, as it tends to give an unrealistic tail approximation.

# Examples
```julia
# discretize a child node at edges -1, 0, 1, approximating tails with spread 1.5:
disc = ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5)
C = ContinuousNode(:C, [:W], disc)
```
"""
struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    sigma::Real

    function ApproximatedDiscretization(intervals::Vector{<:Real}, sigma::Real)
        if !issorted(intervals)
            error("Invalid ApproximatedDiscretization: interval values $intervals are not sorted")
        elseif sigma < 0
            error("Invalid ApproximatedDiscretization: variance must be positive")
        elseif sigma > 2
            @warn "Selected variance values $sigma could be too large for a realistic tails approximation"
        end
        new(intervals, sigma)
    end
end

Base.isempty(d::AbstractDiscretization) = isempty(d.intervals)

ApproximatedDiscretization() = ApproximatedDiscretization(Vector{Real}(), 0)

include("../utils/wrap.jl")
include("../utils/flat.jl")
include("scenariostable.jl")
include("discretenodes.jl")
include("continuousnodes.jl")
include("functionalnodes.jl")