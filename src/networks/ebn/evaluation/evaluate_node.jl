function evaluate(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode, collect::Bool=true)
    scs = simulation_scenarios(node)
    inputs_vector = map(sc -> (sc, simulation_inputs(net, node, sc)), scs)
    if collect
        rt = ScenariosTable{Any}(discrete_ancestors(net, node), :res)
    else
        rt = nothing
    end
    new_continuous = ContinuousNode(node.name, discrete_ancestors(net, node), node.discretization, rt)
    for i in inputs_vector
        scenario = i[1]
        uqinputs = i[2]
        if !UncertaintyQuantification.isimprecise(uqinputs)
            samples = UncertaintyQuantification.sample(uqinputs, node.simulation[scenario...])
            UncertaintyQuantification.evaluate!(node.models, samples)
            new_continuous[scenario...] = EmpiricalDistribution(samples[:, node.models[end].name], node.nbins)
            if collect
                new_continuous.results[scenario...] = samples
            end
        else
            samples = UncertaintyQuantification.sample(uqinputs, MonteCarlo(100))
            UncertaintyQuantification.propagate_intervals!(node.models, samples)
            lbs = map(s -> s.lb, samples[:, node.name])
            ubs = map(s -> s.ub, samples[:, node.name])
            lb_pdf = EmpiricalDistribution(lbs, node.nbins)
            ub_pdf = EmpiricalDistribution(ubs, node.nbins)
            new_continuous[scenario...] = [:lb => lb_pdf, :ub => ub_pdf]
            if collect
                new_continuous.results[scenario...] = samples
            end
        end
    end
    return new_continuous
end

function evaluate(net::EnhancedBayesianNetwork, node::DiscreteFunctionalNode, collect::Bool=true)
    scs = simulation_scenarios(node)
    inputs_vector = map(sc -> (sc, EnhancedBayesianNetworks.simulation_inputs(net, node, sc)), scs)
    if collect
        rt = ScenariosTable{Any}(discrete_ancestors(net, node), :res)
    else
        rt = nothing
    end
    new_discrete = DiscreteNode(node.name, discrete_ancestors(net, node), node.parameters, rt)
    for i in inputs_vector
        scenario = i[1]
        uqinputs = i[2]
        res = probability_of_failure(node.models, node.performance, uqinputs, node.simulation[scenario...])
        if collect
            new_discrete.results[scenario...] = res[2:end]
        end
        new_discrete[(scenario..., node.name => Symbol(string(node.name) * "_failed"))...] = res[1]
        if isa(res[1], Interval)
            new_discrete[(scenario..., node.name => Symbol(string(node.name) * "_safe"))...] = Interval(1 - res[1].ub, 1 - res[1].lb)
        else
            new_discrete[(scenario..., node.name => Symbol(string(node.name) * "_safe"))...] = 1 - res[1]
        end
    end
    return new_discrete
end

function simulation_scenarios(node::FunctionalNode)
    df = node.simulation.data[:, Not("sim")]
    cols = Symbol.(names(df))
    scs = map(row -> Dict(col => row[col] for col in cols), eachrow(df))
    if isempty(scs)
        scs = [Evidence()]
    end
    return scs
end

function simulation_inputs(net::EnhancedBayesianNetwork, node::FunctionalNode, sc::Evidence)
    par_names = parents(net, node)
    par_nodes = filter(n -> n.name ∈ par_names, net.nodes)
    return mapreduce(p -> _inputs(p, Dict(sc)), vcat, par_nodes)
end