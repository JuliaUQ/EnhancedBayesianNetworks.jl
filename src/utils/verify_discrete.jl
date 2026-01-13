function _verify_probabilities!(cpt::DiscreteConditionalProbabilityTable, name)
    if isprecise(cpt)
        probabilities = vcat(cpt.data[!, :Π]...)
        _verify_probability_value.(probabilities)
    elseif !isprecise(cpt)
        probabilities = collect(Iterators.flatten(cpt.data[:, :Π]))
        _verify_probability_value.(probabilities)
        if any(first.(cpt.data[!, :Π]) .- last.(cpt.data[!, :Π]) .> 0)
            error("interval probabilities must have a lower bound smaller than upper bound. $(cpt.data)")
        end
    end
    sub_cpts = _scenarios_cpt(cpt, name)
    cpt.data.Π = mapreduce(sc -> _verify_single_scenario!(sc), vcat, sub_cpts)
end

function _verify_probability_value(p)
    if p < 0
        error("probabilities must be non-negative: $p")
    elseif p > 1
        error("probabilities must be lower or equal than 1: $p")
    end
end

function _verify_single_scenario!(sub_cpt::DataFrame)
    if isa(sub_cpt.Π, Vector{<:PreciseDiscreteProbability})
        total_probability = sum(sub_cpt[!, :Π])
        if total_probability != 1
            if isapprox(total_probability, 1; atol=0.05)
                @warn "total probability should be one, but the evaluated value is $total_probability, and will be normalized"
                sub_cpt[!, :Π] = normalize(sub_cpt[!, :Π], 1)
            else
                error("states are not exhaustive and mutually exclusives for the following cpt: $sub_cpt")
            end
        else
            sub_cpt[!, :Π] = sub_cpt[!, :Π]
        end
    elseif isa(sub_cpt.Π, Vector{<:ImpreciseDiscreteProbability})
        if sum(first.(sub_cpt[!, :Π])) > 1
            error("sum of intervals lower bounds is bigger than 1: $sub_cpt")
        elseif sum(last.(sub_cpt[!, :Π])) < 1
            error("sum of intervals upper bounds is smaller than 1: $sub_cpt")
        end
        sub_cpt[!, :Π] = sub_cpt[!, :Π]
    end
end

function _verify_parameters(cpt::AbstractConditionalProbabilityTable, parameters::Dict{Symbol,Vector{Parameter}}, name::Symbol)
    if !isempty(parameters)
        if !issetequal(states(cpt, name), keys(parameters))
            error("parameters keys $(keys(parameters)) must be coherent with states $(states(cpt, name))")
        end
    end
end