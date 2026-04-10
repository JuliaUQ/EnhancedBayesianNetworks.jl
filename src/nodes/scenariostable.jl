const ContinuousProbability = Union{UnivariateDistribution,ProbabilityBox,Interval,Vector{Pair{Symbol,EmpiricalDistribution}}}
const DiscreteProbability = Union{Real,Interval}
const Probability = Union{ContinuousProbability,DiscreteProbability}

const ContinuousSimulation = AbstractMonteCarlo
const DiscreteSimulation = Union{AbstractSimulation,DoubleLoop,RandomSlicing}
const Simulation = Union{ContinuousSimulation,DiscreteSimulation}

# ScenarioTable is the basic constructur for ConditionalProbabilityTable and SimulationTable
struct ScenariosTable{T<:Union{<:Probability,<:Simulation,Any}}
    data::DataFrame
    n::Symbol
    function ScenariosTable{T}(columns::Union{Symbol,Vector{Symbol}}, n::Symbol) where {T<:Union{<:Probability,<:Simulation,Any}}
        columns = wrap(columns)
        data = DataFrame([col => Symbol[] for col in columns])
        data[:, n] = T[]
        new{T}(data, n)
    end
end

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

function Base.filter(st::ScenariosTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(st.data, selector, view=true)
end

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