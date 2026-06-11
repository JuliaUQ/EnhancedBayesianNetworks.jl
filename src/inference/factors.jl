@auto_hash_equals mutable struct Factor
    dimensions::Vector{Symbol}
    potential::Array
    states_mapping::Dict{Symbol,Dict{Symbol,Int}}

    function Factor(dims::Vector{Symbol}, potential::Array, states_mapping::Dict{Symbol,Dict{Symbol,Int}})
        _check_dims_unique(dims)
        (length(dims) != ndims(potential)) && error("potential must have as many dimensions as length of dimensions")
        (:potential in dims) && error("Having a dimension called potential will cause problems")
        Set(dims) != Set(keys(states_mapping)) && error("states mapping keys have to be coherent with defined dimensions")
        return new(dims, potential, states_mapping)
    end
end

function factorize(cpt::DataFrame)
    cols = Symbol.(names(cpt))
    neworder = [cols[end-1], cols[1:end-2]..., cols[end]]
    new_cpt = select(cpt, neworder)
    result = map(
        (name, col) -> (
            name,
            Dict(v => i for (i, v) in enumerate(unique(col)))
        ),
        Symbol.(names(new_cpt[:, Not(:Π)])),
        eachcol(new_cpt[:, Not(:Π)])
    )

    result = map((name, col) -> (name, unique(col)), Symbol.(names(new_cpt[:, Not(:Π)])), eachcol(new_cpt[:, Not(:Π)]))
    function sts_map_dict(n::Symbol)
        dictionary = Dict{Symbol,Int}()
        for (i, p) in enumerate(unique(new_cpt[!, n]))
            dictionary[p] = i
        end
        return dictionary
    end
    sts = Dict(map(n -> (n => sts_map_dict(n)), node_names))
    dims = map(name -> length(unique(new_cpt[!, name])), node_names)
    potentials = reshape(new_cpt[!, :Π], Tuple(dims))
    return Factor(node_names, potentials, sts)
end

# function factorize(cpt::AbstractConditionalProbabilityTable)
#     factorize(cpt.data)
# end

function _check_dims_valid(dims::Vector{Symbol}, ϕ::Factor)
    isempty(dims) && return
    dim = first(dims)
    (dim in ϕ) || error("Dimension is not in the factor")
    return _check_dims_valid(dims[2:end], ϕ)
end

# dims are unique
_check_dims_unique(dims::Vector{Symbol}) = allunique(dims) || error("Dimensions must be unique")

# # Create a factor for a node, given some evidence.
# function Factor(node::AbstractNode, e::Evidence=Evidence())
#     ϕ = factorize(node.cpt)
#     return ϕ[e]
# end
# function Factor(bn, node::Symbol, e::Evidence=Evidence())
#     node = bn.nodes[bn.topology[node]]
#     return Factor(node, e)
# end

# # reduce the dimension and then squeeze them out
# Base.sum(ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}) = _reducedim(+, ϕ, dims)
# Base.convert(::Type{Factor}, cpt::T) where {T<:AbstractConditionalProbabilityTable} = factorize(cpt)
# Base.size(ϕ::Factor) = size(ϕ.potential)
# Base.size(ϕ::Factor, dim::Symbol) = size(ϕ.potential, indexin(dim, ϕ))
# Base.names(ϕ::Factor) = ϕ.dimensions
# Base.in(dim::Symbol, ϕ::Factor) = dim in names(ϕ)
# Base.indexin(dim::Symbol, ϕ::Factor) = findnext(isequal(dim), ϕ.dimensions, 1)
# Base.indexin(dims::Vector{Symbol}, ϕ::Factor) = indexin(dims, names(ϕ))
# Base.ndims(ϕ::Factor) = ndims(ϕ.potential)
# Base.length(ϕ::Factor) = length(ϕ.potential)

# function Base.getindex(ϕ::Factor, e::Evidence)
#     inds = _translate_index(ϕ, e)
#     keep = inds .== Colon()
#     new_dims = ϕ.dimensions[keep]
#     @inbounds new_p = ϕ.potential[inds...]

#     if ndims(new_p) == 0
#         new_p = dropdims([new_p], dims=1)
#     end

#     states_mapping_dict = filter(((k, v),) -> k ∈ new_dims, ϕ.states_mapping)
#     return Factor(new_dims, new_p, states_mapping_dict)
# end

# Base.getindex(ϕ::Factor, pair::Pair{Symbol}...) = Base.getindex(ϕ, Evidence(pair))

# function Base.join(op, ϕ1::Factor, ϕ2::Factor)
#     if length(ϕ1) < length(ϕ2)
#         ϕ2, ϕ1 = ϕ1, ϕ2
#     end
#     common = intersect(ϕ1.dimensions, ϕ2.dimensions)
#     index_common1 = indexin(common, ϕ1.dimensions)
#     index_common2 = indexin(common, ϕ2.dimensions)
#     if [size(ϕ1)[index_common1]...] != [size(ϕ2)[index_common2]...]
#         error("Common dimensions must have same size")
#     end
#     # the first dimensions are all from ϕ1
#     new_dims = union(ϕ1.dimensions, ϕ2.dimensions)
#     new_states_mapping = merge(ϕ1.states_mapping, ϕ2.states_mapping)
#     # permute the common dimensions in ϕ2 to the beginning,
#     #  in the order that they appear in ϕ1 (and therefore new_dims)            
#     unique1 = setdiff(ϕ1.dimensions, common)
#     unique2 = setdiff(ϕ2.dimensions, common)
#     # these will also be the same indices for new_dims
#     index_unique1 = indexin(unique1, ϕ1.dimensions)
#     index_unique2 = indexin(unique2, ϕ2.dimensions)
#     perm = vcat(index_common2, index_unique2)
#     temp = permutedims(ϕ2.potential, perm)
#     # reshape by lining up the common dims in ϕ2 with those in ϕ1
#     size_unique2 = size(ϕ2)[index_unique2]
#     # set those dims to have dimension 1 for data in ϕ2
#     reshape_lengths = vcat(size(ϕ1)..., size_unique2...)
#     new_v = Array{Float64}(undef, reshape_lengths...)
#     reshape_lengths[index_unique1] .= 1
#     temp = reshape(temp, (reshape_lengths...,))
#     broadcast!(op, new_v, ϕ1.potential, temp)
#     return Factor(new_dims, new_v, new_states_mapping)
# end

# Base.broadcast(f, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, values) = broadcast!(f, deepcopy(ϕ), dims, values)

# function Base.broadcast!(f, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, values)
#     if isa(dims, Symbol)
#         dims = [dims]
#         values = [values]
#     end
#     _check_dims_unique(dims)
#     _check_dims_valid(dims, ϕ)

#     (length(dims) != length(values)) &&
#         error("Number of dimensions does not " * "match number of values to broadcast")

#     inds = indexin(dims, ϕ)

#     reshape_dims = ones(Int, ndims(ϕ))
#     new_values = Vector{Array{Float64}}(undef, length(values))

#     for (i, val) in enumerate(values)
#         if isa(val, Vector{Float64})
#             # reshape to the proper dimension
#             dim_loc = inds[i]
#             @inbounds reshape_dims[dim_loc] = length(val)
#             new_values[i] = reshape(val, reshape_dims...)
#             @inbounds reshape_dims[dim_loc] = 1
#         elseif isa(val, Float64)
#             new_values[i] = [val]
#         else
#             throw(TypeError(:broadcast!, "Invalid broadcast value",
#                 Union{Float64,Vector{Float64}}, val))
#         end
#     end

#     broadcast!(f, ϕ.potential, ϕ.potential, new_values...)

#     return ϕ
# end

# @inline function Base.permutedims!(ϕ::Factor, perm)
#     ϕ.potential = permutedims(ϕ.potential, perm)
#     ϕ.dimensions = ϕ.dimensions[perm]
#     return ϕ
# end

# Base.permutedims(ϕ::Factor, perm) = permutedims!(deepcopy(ϕ), perm)

# *(ϕ1::Factor, ϕ2::Factor) = join(*, ϕ1, ϕ2)

# @inline function _translate_index(ϕ::Factor, e::Evidence)
#     inds = Array{Any}(undef, length(ϕ.dimensions))
#     inds[:] .= Colon()

#     for (i, dim) in enumerate(ϕ.dimensions)
#         if haskey(e, dim)
#             ind = ϕ.states_mapping[dim][e[dim]]
#             inds[i] = ind
#         end
#     end
#     return inds
# end

# function _reducedim(op, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, v0=nothing)
#     dims = wrap(dims)
#     _check_dims_valid(dims, ϕ)
#     # needs to be a tuple for squeeze
#     inds = (indexin(dims, ϕ)...,)

#     dims_new = deepcopy(ϕ.dimensions)
#     deleteat!(dims_new, inds)

#     v_new = _reddim(op, ϕ, inds, v0)
#     states_new = Dict(k => v for (k, v) in ϕ.states_mapping if k ∉ dims)

#     ϕ = Factor(dims_new, v_new, states_new)
#     return ϕ
# end

# function _reducedim!(op, ϕ::Factor, dims::Union{Symbol,Vector{Symbol}}, v0=nothing)
#     dims = wrap(dims)
#     _check_dims_valid(dims, ϕ)
#     # needs to be a tuple for squeeze
#     inds = (indexin(dims, ϕ)...,)
#     deleteat!(ϕ.dimensions, inds)
#     ϕ.potential = _reddim(op, ϕ, inds, v0)
#     ϕ.states_mapping = Dict(k => v for (k, v) in ϕ.states_mapping if k ∉ dims)
#     return ϕ
# end

# _reddim(op, ϕ::Factor, inds::Tuple, ::Nothing) =
#     dropdims(reduce(op, ϕ.potential, dims=inds), dims=inds)