const ContinuousSimulation = AbstractMonteCarlo
const DiscreteSimulation = Union{AbstractSimulation,DoubleLoop,RandomSlicing}
const Simulation = Union{ContinuousSimulation,DiscreteSimulation}

struct SimulationTable{T<:Union{ContinuousSimulation,DiscreteSimulation}}
    data::DataFrame
    function SimulationTable{T}(columns::Union{Symbol,Vector{Symbol}}) where {T<:Union{ContinuousSimulation,DiscreteSimulation}}
        columns = wrap(columns)
        data = DataFrame([col => Symbol[] for col in columns])
        data[:, :sim] = T[]
        new{T}(data)
    end
end

function Base.setindex!(st::SimulationTable, value::Simulation, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    evidence_nodes = collect(map(p -> p[1], key))
    st_nodes = Symbol.(filter(i -> i != "sim", names(st.data)))
    if issetequal(evidence_nodes, st_nodes)
        cp = subset(st.data, selector, view=true)
        if isempty(cp)
            push!(st.data, (key..., sim=value))
        else
            @assert size(cp, 1) == 1
            cp.sim[1] = value
        end
    else
        error("Cannot set index with $evidence_nodes into a SimulationTable initialized with $st_nodes")
    end
end

function Base.getindex(st::SimulationTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    cp = subset(st.data, selector, view=true)
    if isempty(cp)
        error("index not find in the SimlationTable $st")
    else
        @assert size(cp, 1) == 1
        return cp.sim[1]
    end
end

function Base.filter(st::SimulationTable, key::Pair{Symbol,Symbol}...)
    selector = map((p) -> p[1] => ByRow(x -> x == p[2]), collect(key))
    return subset(st.data, selector, view=true)
end