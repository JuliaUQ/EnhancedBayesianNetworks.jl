function evaluate(net::EnhancedBayesianNetwork, node::ContinuousFunctionalNode; nbins::Int=0)
    scs = map(row -> [Symbol(col) => row[col] for col in names(node.simulation.data[:, Not("sim")])], eachrow(node.simulation.data[:, Not("sim")]))
    inputs_vector = map(sc -> (sc, simulation_inputs(net, node, sc)), scs)
    rt = ResultTable(discrete_ancestors(net, node))
    new_continuous = ContinuousNode(node.name, discrete_ancestors(net, node), node.discretization, rt)
    for i in inputs_vector
        scenario = i[1]
        uqinputs = i[2]
        if !UncertaintyQuantification.isimprecise(uqinputs)
            samples = UncertaintyQuantification.sample(uqinputs, node.simulation[scenario...])
            UncertaintyQuantification.evaluate!(node.models, samples)
            new_continuous[scenario...] = EmpiricalDistribution(samples[:, node.models[end].name], node.nbins)
            new_continuous.results[scenario...] = samples
        else
            samples = UncertaintyQuantification.sample(uqinputs, MonteCarlo(100))
            UncertaintyQuantification.propagate_intervals!(node.models, samples)
            lbs = map(s -> s.lb, samples[:, node.name])
            ubs = map(s -> s.ub, samples[:, node.name])
            lb_pdf = EmpiricalDistribution(lbs, node.nbins)
            ub_pdf = EmpiricalDistribution(ubs, node.nbins)
            new_continuous[scenario...] = [:lb => lb_pdf, :ub => ub_pdf]
            new_continuous.results[scenario...] = samples
        end
    end
    return new_continuous
end

function evaluate(net::EnhancedBayesianNetwork, node::DiscreteFunctionalNode)
    scs = map(row -> [Symbol(col) => row[col] for col in names(node.simulation.data[:, Not("sim")])], eachrow(node.simulation.data[:, Not("sim")]))
    inputs_vector = map(sc -> (sc, simulation_inputs(net, node, sc)), scs)
    rt = ResultTable(discrete_ancestors(net, node))
    new_discrete = DiscreteNode(node.name, discrete_ancestors(net, node), node.parameters, rt)
    for i in inputs_vector
        scenario = i[1]
        uqinputs = i[2]
        res = probability_of_failure(node.models, node.performance, uqinputs, node.simulation[scenario...])
        new_discrete.results[scenario...] = res[2:end]
        new_discrete[vcat(scenario, node.name => Symbol(string(node.name) * "_failed"))...] = res[1]
        if isa(res[1], Interval)
            new_discrete[vcat(scenario, node.name => Symbol(string(node.name) * "_safe"))...] = Interval(1 - res[1].ub, 1 - res[1].lb)
        else
            new_discrete[vcat(scenario, node.name => Symbol(string(node.name) * "_safe"))...] = 1 - res[1]
        end
    end
    return new_discrete
end

function simulation_inputs(net::EnhancedBayesianNetwork, node::FunctionalNode, sc::Vector{Pair{Symbol,Symbol}})
    par_names = parents(net, node)
    par_nodes = filter(n -> n.name ∈ par_names, net.nodes)
    uqinputs = mapreduce(p -> EnhancedBayesianNetworks._inputs(p, Dict(sc)), vcat, par_nodes)
    return uqinputs
end