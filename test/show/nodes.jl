@testsnippet SetupShowNet begin
    using EnhancedBayesianNetworks: _show_parents, _show_discretization, _show_parameters, _show_models, _pq_string, _topology_parents, ExactDiscretization, ApproximatedDiscretization

    # capture a type's full `show(io, ::MIME"text/plain", x)` and a bare helper's output
    plainshow(x) = sprint((io, y) -> show(io, MIME"text/plain"(), y), x)
    helper(f, args...) = sprint(f, args...)

    # ---- sample nodes: precise/credal discrete, discretized/plain continuous, both functionals ----
    W = DiscreteNode(:W)
    W[:W=>:sunny] = 0.7
    W[:W=>:cloudy] = 0.3

    S = DiscreteNode(:S, [:W])
    S[:W=>:sunny, :S=>:on] = 0.6
    S[:W=>:sunny, :S=>:off] = 0.4
    S[:W=>:cloudy, :S=>:on] = 0.2
    S[:W=>:cloudy, :S=>:off] = 0.8

    C = DiscreteNode(:C)
    C[:C=>:yes] = Interval(0.2, 0.4)
    C[:C=>:no] = Interval(0.6, 0.8)

    Sp = DiscreteNode(:Sp, [:on => [Parameter(0.5, :Sp)], :off => [Parameter(0.0, :Sp)]])
    Sp[:Sp=>:on] = 0.6
    Sp[:Sp=>:off] = 0.4

    Rc = ContinuousNode(:Rc, ExactDiscretization([-1.0, 1.0]))
    Rc[] = Normal()

    U = ContinuousNode(:U, [:W], ApproximatedDiscretization([-1.0, 1.0], 2))
    U[:W=>:sunny] = Normal()
    U[:W=>:cloudy] = Normal(2, 1)

    D = ContinuousNode(:D)
    D[] = Normal(1, 2)

    cf = ContinuousFunctionalNode(:fc, [Model(df -> df.D .^ 2, :fc)], MonteCarlo(100))
    fd = DiscreteFunctionalNode(:fd, [Model(df -> df.fc .+ 1, :G)], df -> df.G .- 1.0, MonteCarlo(100))

    # ---- EBN (D -> fc) ----
    ebn = EnhancedBayesianNetwork([D, cf])
    add_child!(ebn, D, cf)
    order!(ebn)

    # ---- BN (W -> S -> R) ----
    Wb = DiscreteNode(:W)
    Wb[:W=>:sunny] = 0.7
    Wb[:W=>:cloudy] = 0.3
    Sb = DiscreteNode(:S, [:W])
    Sb[:W=>:sunny, :S=>:on] = 0.6
    Sb[:W=>:sunny, :S=>:off] = 0.4
    Sb[:W=>:cloudy, :S=>:on] = 0.2
    Sb[:W=>:cloudy, :S=>:off] = 0.8
    Rb = DiscreteNode(:R, [:S])
    Rb[:S=>:on, :R=>:hi] = 0.5
    Rb[:S=>:on, :R=>:lo] = 0.5
    Rb[:S=>:off, :R=>:hi] = 0.1
    Rb[:S=>:off, :R=>:lo] = 0.9
    bn = BayesianNetwork([Wb, Sb, Rb])
    add_child!(bn, :W, :S)
    add_child!(bn, :S, :R)
    order!(bn)

    # ---- CN (Tampering, Fire -> Alarm) ----
    Tc = DiscreteNode(:Tampering)
    Tc[:Tampering=>:YesT] = 0.98
    Tc[:Tampering=>:NoT] = 0.02
    Fc = DiscreteNode(:Fire)
    Fc[:Fire=>:YesF] = Interval(0.98, 0.99)
    Fc[:Fire=>:NoF] = Interval(0.01, 0.02)
    Ac = DiscreteNode(:Alarm, [:Tampering, :Fire])
    Ac[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.4, 0.6)
    Ac[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.4, 0.5)
    Ac[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.85, 0.9)
    Ac[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.1, 0.15)
    Ac[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.985, 0.99)
    Ac[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.01, 0.015)
    Ac[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.0001, 0.0002)
    Ac[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.9998, 0.9999)
    cn = CredalNetwork([Tc, Fc, Ac])
    add_child!(cn, :Tampering, :Alarm)
    add_child!(cn, :Fire, :Alarm)
    order!(cn)

    # ---- posteriors built directly (deterministic, no inference/randomness) ----
    ns = EnhancedBayesianNetworks.NetworkSchema(
        Dict(:W => 1),
        [:W],
        [Dict(:Cloudy => 1, :Sunny => 2)],
        [[:Cloudy, :Sunny]]
    )
    f1 = EnhancedBayesianNetworks.Factor([1], [0.4, 0.6])
    f2 = EnhancedBayesianNetworks.Factor([1], [0.3, 0.7])
    post = EnhancedBayesianNetworks.Posterior(f1, ns, [:W], EnhancedBayesianNetworks.Evidence())
    poste = EnhancedBayesianNetworks.Posterior(f1, ns, [:W], EnhancedBayesianNetworks.Evidence(:X => :on))
    cpost = EnhancedBayesianNetworks.CredalPosterior([post], f2, f1, ns, [:W], EnhancedBayesianNetworks.Evidence())
end

@testitem "Show - node helpers" setup = [SetupShowNet] begin
    @test helper(_show_parents, W) == "Parents: none\n"
    @test helper(_show_parents, S) == "Parents: W\n"
    @test helper(_show_discretization, ExactDiscretization([-1.0, 1.0])) ==
          "Discretization: ExactDiscretization\n  Intervals: -1.0, 1.0\n"
    @test helper(_show_discretization, ApproximatedDiscretization([-1.0, 1.0], 2)) ==
          "Discretization: ApproximatedDiscretization\n  Sigma: 2\n  Intervals: -1.0, 1.0\n"
    @test helper(_show_discretization, D.discretization) == ""
    @test helper(_show_models, cf.models) == "Models: 1\n  Names: fc\n"
    @test helper(_show_parameters, W.parameters) == ""
    @test helper(_show_parameters, Sp.parameters) ==
          "Parameters:\n  on: UncertaintyQuantification.Parameter(0.5, :Sp)\n  off: UncertaintyQuantification.Parameter(0.0, :Sp)\n"
end

@testitem "Show - DiscreteNode" setup = [SetupShowNet] begin
    @test sprint(show, W) == "DiscreteNode(W, parents=Symbol[], states=[:sunny, :cloudy])"
    @test sprint(show, S) == "DiscreteNode(S, parents=[:W], states=[:on, :off])"
    @test sprint(show, C) == "DiscreteNode(C, parents=Symbol[], states=[:yes, :no])"
    w = plainshow(W)
    @test occursin("DiscreteNode: W", w)
    @test occursin("Parents: none", w)
    @test occursin("States: sunny, cloudy", w)
    @test occursin("Type: Precise", w)
    @test occursin("DataFrame", w)
    @test occursin("Type: Credal", plainshow(C))
    @test occursin("Parameters:", plainshow(Sp))
end

@testitem "Show - ContinuousNode" setup = [SetupShowNet] begin
    @test sprint(show, Rc) == "ContinuousNode(Rc, parents=Symbol[], discretization=ExactDiscretization)"
    @test sprint(show, U) == "ContinuousNode(U, parents=[:W], discretization=ApproximatedDiscretization)"
    @test sprint(show, D) == "ContinuousNode(D, parents=Symbol[], discretization=ExactDiscretization)"
    u = plainshow(U)
    @test occursin("ContinuousNode: U", u)
    @test occursin("Parents: W", u)
    @test occursin("Discretization: ApproximatedDiscretization", u)
    @test occursin("Sigma: 2", u)
    @test occursin("Type: Precise", u)
    @test occursin("Support:", u)
    d = plainshow(D)
    @test occursin("ContinuousNode: D", d)
    @test !occursin("Discretization:", d)
end

@testitem "Show - functional nodes" setup = [SetupShowNet] begin
    @test sprint(show, cf) == "ContinuousFunctionalNode(fc, models=1, nbins=0)"
    @test sprint(show, fd) == "DiscreteFunctionalNode(fd, states=[:fd_safe, :fd_failed], models=1)"
    c = plainshow(cf)
    @test occursin("ContinuousFunctionalNode: fc", c)
    @test occursin("Models: 1", c)
    @test occursin("Bins: 0", c)
    @test occursin("Simulation: MonteCarlo", c)
    f = plainshow(fd)
    @test occursin("DiscreteFunctionalNode: fd", f)
    @test occursin("States: fd_safe, fd_failed", f)
    @test occursin("Simulation: MonteCarlo", f)
end