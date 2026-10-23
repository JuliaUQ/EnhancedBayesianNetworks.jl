@testitem "DirectAcyclicGraph construction" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    @test isempty(dag.nodes)
    @test isempty(dag.topology)
    @test size(dag.A) == (0, 0)
    @test isempty(dag.states)
    @test isempty(dag.estimators)

    # add_node! returns the DAG itself, so declarations can be chained or piped
    @test add_node!(dag, :W) === dag
    @test [n.name for n in dag.nodes] == [:W]
    @test dag.topology == Dict(:W => 1)
    @test size(dag.A) == (1, 1)

    # the adjacency matrix grows by one row and column per node, and stays square
    add_node!(dag, :R; parents = [:W], states = [:yesR])
    add_node!(dag, :G; parents = [:R], states = [:maybeG], estimator = PreciseDirichletModel())
    @test size(dag.A) == (3, 3)
    @test dag.topology == Dict(:W => 1, :R => 2, :G => 3)
end

@testitem "DirectAcyclicGraph add_node!" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()
    add_node!(dag, :W)
    add_node!(dag, :R; parents = [:W])

    @test_throws ErrorException("Invalid DAG: node :W is already present") add_node!(dag, :W)
    @test_throws ErrorException("Invalid DAG: parent(s) [:NOPE] of :X are not defined; add them first") add_node!(dag, :X; parents = [:NOPE])

    # a node cannot be its own parent: it is not in the DAG yet when its parents are checked, so the undefined-parent check catches the cycle before any edge is wired
    @test_throws ErrorException("Invalid DAG: parent(s) [:S] of :S are not defined; add them first") add_node!(dag, :S; parents = [:S])

    # a parent listed twice is rejected while the node's CPT is built, not by the DAG itself, so the  error comes from the table rather than carrying the "Invalid DAG" prefix of the checks above
    @test_throws ArgumentError add_node!(dag, :Z; parents = [:W, :W])

    # every rejection above leaves the DAG exactly as it was: the checks run before `push!`, and the  CPT is built before it, so a half-added node is never observable
    @test [n.name for n in dag.nodes] == [:W, :R]
    @test dag.topology == Dict(:W => 1, :R => 2)
    @test size(dag.A) == (2, 2)
end

@testitem "DAG parents, children and adjacency" setup = [ExtraDeps] begin
    using SparseArrays

    # :P is inserted before :R, but :G declares its parents in the opposite order
    dag = DirectAcyclicGraph()
    add_node!(dag, :W)
    add_node!(dag, :P; parents = [:W])
    add_node!(dag, :R; parents = [:W])
    add_node!(dag, :G; parents = [:R, :P])

    # `parents` reads the adjacency matrix column, so it returns NODE INSERTION ORDER, not the order the parents were declared in. `_fit_cpts` builds each CPT key prefix from this, so the order is part of the contract rather than an accident
    @test parents(dag, :G) == [:P, :R]
    @test parents(dag, :W) == Symbol[]
    @test children(dag, :W) == [:P, :R]
    @test children(dag, :G) == Symbol[]

    # one edge per declared parent, and nothing else
    rows, cols, _ = findnz(dag.A)
    @test Set(zip(rows, cols)) == Set([(1, 2), (1, 3), (3, 4), (2, 4)])
    @test eltype(dag.A) == Bool
end

@testitem "DAG declared states and estimators" setup = [ExtraDeps] begin
    dag = DirectAcyclicGraph()

    # omitted keywords store nothing, so `_verify_estimators` can tell "declared none" from "declared an empty list" and `_domains` can take the whole domain from the data
    add_node!(dag, :W)
    @test !haskey(dag.states, :W)
    @test !haskey(dag.estimators, :W)

    # an explicitly empty vector is still nothing declared
    add_node!(dag, :P; states = Symbol[], parents = [:W])
    @test !haskey(dag.states, :P)
    add_node!(dag, :Q; parents = [:W], estimator = nothing)
    @test !haskey(dag.estimators, :Q)

    # declared states are kept verbatim, unsorted: `_domains` sorts them together with the states observed in the data, so the DAG itself never reorders what was declared
    add_node!(dag, :R; states = [:b, :a], parents = [:W], estimator = ImpreciseDirichletModel(2))
    @test dag.states[:R] == [:b, :a]
    @test dag.estimators[:R] == ImpreciseDirichletModel(2)

    # one DAG may mix precise and imprecise declarations; `learn` is what rejects a partial set
    add_node!(dag, :G; parents = [:R, :P], estimator = PreciseDirichletModel(1))
    @test dag.estimators[:G] == PreciseDirichletModel(1)
    @test sort(collect(keys(dag.estimators))) == [:G, :R]
end
