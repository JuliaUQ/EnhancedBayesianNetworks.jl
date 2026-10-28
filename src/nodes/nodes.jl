# Node type hierarchy: every node is discrete- or continuous-flavoured;
abstract type AbstractNode end
abstract type AbstractContinuousNode <: AbstractNode end
abstract type AbstractDiscreteNode <: AbstractNode end

# Discretization strategy attached to a continuous node (see ExactDiscretization / ApproximatedDiscretization).
abstract type AbstractDiscretization end

# Default terminal mesh size of the p-box interval-mass optimisation, and the accepted `technique` values for that mass. UQ uses 1e-13 for imprecise failure probabilities; a probability mass needs far less, and the cost grows steeply with it.
const PBOX_MASS_MESH = 1.0e-8
const PBOX_MASS_TECHNIQUES = (:optimised, :distribution_free)

# Shared validation of the two p-box mass settings carried by every discretization strategy.
function _check_pbox_mass(mesh::Real, technique::Symbol)
    if mesh <= 0
        error("Invalid discretization: p-box mass mesh must be positive, got $mesh")
    elseif technique ∉ PBOX_MASS_TECHNIQUES
        error("Invalid discretization: unknown p-box mass technique $(repr(technique)), must be one of $(join(repr.(PBOX_MASS_TECHNIQUES), ", "))")
    end
    return nothing
end

"""
    ExactDiscretization(intervals = Real[]; mesh = $PBOX_MASS_MESH, technique = :optimised)

Discretization strategy for a continuous **root** node, allowing evidence to be observed on it. The node's distribution support is partitioned exactly at the sorted `intervals` edges, turning the continuous root into discrete bins. The default (empty `intervals`) leaves the node continuous, i.e. no discretization is applied. The edges must be sorted.

`technique` and `mesh` control how the probability mass of a bin is obtained when the node's entry is a `ProbabilityBox`, and are ignored for a precise distribution or an `Interval`:

  - `:optimised` (default) minimises and maximises the mass over the parameter box, one member of the family at a time, with `mesh` as the optimiser's terminal mesh size. Smaller is tighter and slower.
  - `:distribution_free` differences the CDF envelopes instead. Valid for *any* CDF lying between them, hence an enclosure of the parametric answer rather than the answer itself, but far cheaper and the only sound choice once the p-box is not described by a parameter box. `mesh` is unused.

# Examples
```julia
# discretize a root node's support at the edges -2, 0, 2:
disc = ExactDiscretization([-2.0, 0.0, 2.0])
T = ContinuousNode(:T, Normal(), disc)

ExactDiscretization()            # empty: the root stays continuous

# a coarser, cheaper mesh for an imprecise root, or the enclosure instead of the optimisation:
ExactDiscretization([-2.0, 0.0, 2.0]; mesh = 1.0e-4)
ExactDiscretization([-2.0, 0.0, 2.0]; technique = :distribution_free)
```
"""
struct ExactDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    mesh::Real
    technique::Symbol

    function ExactDiscretization(
            intervals::Vector{<:Real};
            mesh::Real = PBOX_MASS_MESH,
            technique::Symbol = :optimised
        )
        if !issorted(intervals)
            error("Invalid ExactDiscretization: interval values $intervals are not sorted")
        end
        _check_pbox_mass(mesh, technique)
        return new(intervals, mesh, technique)
    end
end

ExactDiscretization(; kwargs...) = ExactDiscretization(Vector{Real}(); kwargs...)

"""
    ApproximatedDiscretization(intervals = Real[], λ = 0; mesh = $PBOX_MASS_MESH, technique = :optimised)

Discretization strategy for a continuous **non-root (child)** node, allowing evidence to be observed on it. The sorted `intervals` edges partition the support into discrete bins, while `λ` is the rate parameter of the exponential used to approximate the original continuous distribution's tails when a discrete state is mapped back to a continuous range. `λ` must be non-negative; a value above `2` is accepted but warns, as it tends to give an unrealistic tail approximation.

`technique` and `mesh` are as in [`ExactDiscretization`](@ref): they select how a `ProbabilityBox` entry's bin mass is computed, and have no effect on a precise distribution or an `Interval`.

# Examples
```julia
# discretize a child node at edges -1, 0, 1, approximating tails with rate 1.5:
disc = ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5)
C = ContinuousNode(:C, [:W], disc)

# same edges, but take the CDF-envelope enclosure for any imprecise entry:
ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5; technique = :distribution_free)
```
"""
struct ApproximatedDiscretization <: AbstractDiscretization
    intervals::Vector{<:Real}
    λ::Real
    mesh::Real
    technique::Symbol

    function ApproximatedDiscretization(
            intervals::Vector{<:Real}, λ::Real;
            mesh::Real = PBOX_MASS_MESH,
            technique::Symbol = :optimised
        )
        if !issorted(intervals)
            error("Invalid ApproximatedDiscretization: interval values $intervals are not sorted")
        elseif λ < 0
            error("Invalid ApproximatedDiscretization: rate parameter must be non-negative")
        elseif λ > 2
            @warn "Selected rate parameter $λ could be too large for a realistic tails approximation"
        end
        _check_pbox_mass(mesh, technique)
        return new(intervals, λ, mesh, technique)
    end
end

Base.isempty(d::AbstractDiscretization) = isempty(d.intervals)

ApproximatedDiscretization(; kwargs...) = ApproximatedDiscretization(Vector{Real}(), 0; kwargs...)

include("../utils/wrap.jl")
include("../utils/flat.jl")
include("scenariostable.jl")
include("discretenodes.jl")
include("continuousnodes.jl")
include("functionalnodes.jl")
