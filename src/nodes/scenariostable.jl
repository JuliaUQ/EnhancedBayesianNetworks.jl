# Probability value families: what may fill the value column of a ScenariosTable.
const ContinuousProbability = Union{UnivariateDistribution,ProbabilityBox,Interval,Vector{Pair{Symbol,EmpiricalDistribution}}}
const DiscreteProbability = Union{Real,Interval}
const Probability = Union{ContinuousProbability,DiscreteProbability}

# Simulation families: the strategies a functional node stores per scenario.
const ContinuousSimulation = AbstractMonteCarlo
const DiscreteSimulation = Union{AbstractSimulation,DoubleLoop,RandomSlicing}
const Simulation = Union{ContinuousSimulation,DiscreteSimulation}

# A small table backing both conditional probability tables and simulation tables: a DataFrame with
# one Symbol column per parent (holding states) plus a single value column named `n` (`:Π` for
# probabilities, `:sim` for simulations), typed by `T`.
struct ScenariosTable{T<:Union{<:Probability,<:Simulation,Any}}
    data::DataFrame
    n::Symbol
    # Build an empty table: Symbol columns for each parent, plus an empty T-typed value column `n`.
    function ScenariosTable{T}(columns::Union{Symbol,Vector{Symbol}}, n::Symbol) where {T<:Union{<:Probability,<:Simulation,Any}}
        columns = _wrap(columns)
        data = DataFrame([col => Symbol[] for col in columns])
        data[:, n] = T[]
        new{T}(data, n)
    end
end

# Set (or overwrite) the value for a full parent-state key. 
# Probabilities are range-checked; the key must name exactly the table's parent columns, otherwise it errors. 
# A new row is pushed if the combination is absent, else the existing single matching row is updated.
function Base.setindex!(st::ScenariosTable{T}, value::T, key::Pair{Symbol,Symbol}...) where {T<:Union{<:Probability,<:Simulation,Any}}
    if T == DiscreteProbability
        verify_probability_value(value)
    end
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    st_nodes = filter(i -> i != st.n, propertynames(st.data))
    if issetequal(evidence_nodes, st_nodes)
        cp = subset(st.data, selector, view=true)
        if isempty(cp)
            push!(st.data, (; key..., st.n => value))
        else
            @assert size(cp, 1) == 1
            cp[1, st.n] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a ScenariosTable initialized with $st_nodes")
    end
end

# Look up the single value whose row matches the full parent-state key; errors if absent.
function Base.getindex(st::ScenariosTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(st.data, selector, view=true)
    if isempty(cp)
        error("Index $(collect(key)) not found in the ScenariosTable $st")
    else
        @assert size(cp, 1) == 1
        return cp[1, st.n]
    end
end

# Return (a view of) the sub-rows matching a partial key — e.g. all rows for a given parent state.
function Base.filter(st::ScenariosTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(st.data, selector, view=true)
end

# A probability entry must lie in [0, 1] — checked for both precise Reals and Interval bounds.
function verify_probability_value(value::Real)
    if !(0 <= value <= 1)
        throw(ArgumentError("Probability $value must be >= 0 and <= 1"))
    end
end

function verify_probability_value(value::Interval)
    if !all(0 .<= UncertaintyQuantification.bounds(value) .<= 1)
        throw(ArgumentError("Probability $value must be >= 0 and <= 1"))
    end
end