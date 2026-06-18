function apply_evidence!(factors::Vector{<:Factor}, evidence_idx::Vector{Tuple{Int,Int}})
    for (node, state) in evidence_idx
        for i in eachindex(factors)
            if !containsvar(factors[i], node)
                continue
            end
            factors[i] = restrict(factors[i], node, state)
        end
    end
end

function eliminate_var(factors::Vector{<:Factor}, var::Int)
    involved = similar(factors, 0)
    remaining = similar(factors, 0)
    for f in factors
        if containsvar(f, var)
            push!(involved, f)
        else
            push!(remaining, f)
        end
    end
    if isempty(involved)
        return factors
    end
    push!(remaining, sumout(multiply(involved), var))
    return remaining
end

# function infer(inf::PreciseInferenceState)
#     bn = inf.bn
#     nodes = bn.nodes
#     query = inf.query
#     evidence = inf.evidence
#     factors = map(n -> Factor(bn, n.name, evidence), nodes)
#     # successively remove the hidden nodes
#     δ = [x[1] for x in _order_with_minimal_increase_in_complexity(factors, bn.topology)]
#     δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence))), δ))
#     list = []
#     while !isempty(δ)
#         h = first(δ)
#         push!(list, h)
#         contain_h = filter(ϕ -> h ∈ ϕ, factors)
#         if !isempty(contain_h)
#             factors = setdiff(factors, contain_h)
#             τ_h = sum(reduce((*), contain_h), h)
#             push!(factors, τ_h)
#         end
#         δ = [x[1] for x in _order_with_minimal_increase_in_complexity(factors, bn.topology)]
#         δ = deleteat!(δ, findall(x -> x ∈ vcat(query, collect(keys(evidence)), list), δ))
#     end
#     ϕ = reduce((*), factors)
#     tot = sum(abs, ϕ.potential)
#     ϕ.potential ./= tot
#     return ϕ
# end

# function infer(inf::ImpreciseInferenceState)
#     cn = inf.cn
#     nodes = cn.nodes
#     query = inf.query
#     evidence = inf.evidence

#     dims = length(query) + 1
#     all_nodes = map(node -> _extreme_points(node), nodes)
#     all_nodes_combination = vec(collect(Iterators.product(all_nodes...)))
#     all_nodes_combination = map(t -> [t...], all_nodes_combination)

#     bns = map(anc -> BayesianNetwork(anc, cn.topology, cn.A), all_nodes_combination)

#     r = map(bn -> infer(bn, query, evidence), bns)

#     res = stack(map(r -> r.potential, r))

#     a = minimum(res; dims=dims)
#     b = maximum(res; dims=dims)

#     potential = map((a, b) -> [a, b], a, b)
#     potential = reshape(potential, size(r[1].potential))
#     return Factor(r[1].dimensions, potential, r[1].states_mapping)
# end

# function _order_with_minimal_increase_in_complexity(factors::Vector{Factor}, topology::Dict{Symbol,Int64})
#     dimensions = map(f -> f.dimensions, factors)
#     res = map(x -> (x, _n_added_edges(dimensions, topology, topology[x]) / _n_eliminated_edges(dimensions, topology, topology[x])), collect(keys(topology)))
#     return sort(res, by=x -> x[2])
# end

# function _n_eliminated_edges(dimensions::AbstractVector{Vector{Symbol}}, topology::Dict{Symbol,Int}, index::Int)
#     structure_A = _structure_A(dimensions, topology)
#     return length(structure_A[index, :].nzind)
# end

# function _n_added_edges(dimensions::AbstractVector{Vector{Symbol}}, topology::Dict{Symbol,Int}, index::Int)
#     reverse_dict = Dict(value => key for (key, value) in topology)
#     node = reverse_dict[index]
#     structure_A = _structure_A(dimensions, topology)
#     former_edges = length(structure_A.nzval) - 2 * _n_eliminated_edges(dimensions, topology, index)
#     function _ridimensionalize(d::AbstractVector{Symbol})
#         return filter(x -> x != node, d)
#     end
#     ## Adding the new connection among parents and children
#     new_connection = filter(x -> node ∈ x, dimensions)
#     new_dims = map(dim -> _ridimensionalize(dim), dimensions)
#     if !isempty(new_connection)
#         new_connection = mapreduce(x -> _ridimensionalize(x), vcat, new_connection)
#         push!(new_dims, new_connection)
#     end
#     function _retopologyse(topo::Dict{Symbol,Int})
#         new_dict = Dict{Symbol,Int}()
#         for (k, v) in collect(topo)
#             if k != node
#                 if v > index
#                     new_dict[k] = v - 1
#                 else
#                     new_dict[k] = v
#                 end
#             end
#         end
#         return new_dict
#     end
#     new_topology = _retopologyse(topology)
#     new_structure_A = _structure_A(new_dims, new_topology)
#     return Int((length(new_structure_A.nzval) - former_edges) / 2)
# end

# function _structure_A(dimensions::AbstractVector{Vector{Symbol}}, topology::Dict{Symbol,Int})
#     n = length(topology)
#     structure_A = zeros(n, n)

#     function _structure_link(dim::AbstractVector{Symbol})
#         links = Vector{}()
#         if length(dim) > 1
#             collection = collect(Iterators.product(dim, dim))
#             collection = map(t -> [t...], collection)
#             collection = vec(map(v -> [topology[v[1]], topology[v[2]]], collection))
#             filter!(c -> c[1] != c[2], collection)
#             append!(links, collection)
#         end
#         return links
#     end

#     structural_links = unique!(mapreduce(dim -> _structure_link(dim), vcat, dimensions))
#     for link in structural_links
#         structure_A[link[1], link[2]] = 1
#     end
#     return sparse(structure_A)
# end

# infer(bn::BayesianNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) = infer(PreciseInferenceState(bn, query, evidence))

# infer(cn::CredalNetwork, query::Union{Symbol,Vector{Symbol}}, evidence::Evidence=Evidence()) = infer(ImpreciseInferenceState(cn, query, evidence))