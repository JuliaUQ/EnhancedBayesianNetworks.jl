mutable struct BayesianNetwork <: AbstractNetwork
    nodes::AbstractVector{DiscreteNode}
    topology::Dict
    A::SparseMatrixCSC

    function BayesianNetwork(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
        node_names = map(i -> i.name, nodes)
        dups = not_unique_elements(node_names)
        if !isempty(dups)
            error("Invalid BN: duplicate node names $dups")
        end
        states_list = mapreduce(i -> states(i), vcat, nodes)
        dups = not_unique_elements(states_list)
        if !isempty(dups)
            error("Invalid BN: duplicate node states $dups")
        end
        imprecise_nodes = nodes[.!isprecise.(nodes)]
        if !isempty(imprecise_nodes)
            error("Invalid BN: node/s $(getproperty.(imprecise_nodes, :name)) are imprecise; CrealNetwork structure is required")
        end
        new(nodes, topology, A)
    end
end

function BayesianNetwork(nodes::AbstractVector{DiscreteNode})
    n = length(nodes)
    topology = Dict{Symbol,Int}()
    for (i, n) in enumerate(nodes)
        topology[n.name] = i
    end
    A = spzeros(Bool, n, n)
    return BayesianNetwork(nodes, topology, A)
end

# function joint_probability(bn::BayesianNetwork, scenario::Evidence)
#     th_keys = [i.name for i in bn.nodes]
#     pr_keys = keys(scenario) |> collect
#     if !issubset(th_keys, pr_keys)
#         error("Not all the BN's nodes $([i.name for i in bn.nodes]) have a specidied states in $scenario. Use Inference!")
#     end
#     for k in setdiff(pr_keys, th_keys)
#         @warn("nodes $k is not part of the BN, therefore is useless for the scenario probability evaluation")
#         delete!(scenario, k)
#     end
#     th_states = Dict(map(n -> (n.name, states(n)), bn.nodes))
#     for (node, th_state) in th_states
#         if scenario[node] ∉ th_state
#             error("node $node has a defined scenario state $(scenario[node]) that is not among its possible states $th_state")
#         end
#     end

#     prob = 1.0
#     cpts_dict = Dict(map(n -> (n.name, n.cpt.data), bn.nodes))
#     parents_dict = Dict(map(n -> (n.name, parents(bn, n)[2]), bn.nodes))
#     for (node, cpt) in cpts_dict
#         parent_keys = get(parents_dict, node, [])
#         all_keys = vcat(parent_keys, node)
#         row = filter(r -> all(k -> r[k] == scenario[k], all_keys), eachrow(cpt))
#         prob *= row.Π[1]
#     end
#     return prob
# end