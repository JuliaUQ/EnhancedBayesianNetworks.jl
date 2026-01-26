@testset "Parameters Learning - EM" begin

    df = CSV.read("learning/parameters_learning/sprinkler.csv", DataFrame)
    df_m = CSV.read("learning/parameters_learning/sprinkler_missing.csv", DataFrame)
    bn = BayesianNetwork2be([:Weather, :Sprinkler, :Rain, :Grass])
    add_child!(bn, :Weather, :Sprinkler)
    add_child!(bn, :Weather, :Rain)
    add_child!(bn, :Sprinkler, :Grass)
    add_child!(bn, :Rain, :Grass)
    order!(bn)

    states_space = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry]
    )
    states_space2 = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No, :Pooring],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry]
    )
    states_space3 = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry, :New]
    )

    @testset "Explode missings" begin
        nt1 = copy(df_m[2, :])
        exploded_nt1 = EnhancedBayesianNetworks._explode_missing_single_line(nt1, states_space)
        @test nrow(exploded_nt1) == 2
        @test issetequal(exploded_nt1.Weather, states_space[:Weather])

        nt2 = copy(df_m[3, :])
        exploded_nt2 = EnhancedBayesianNetworks._explode_missing_single_line(nt2, states_space)
        @test nrow(exploded_nt2) == 4
        @test issetequal(exploded_nt2.Rain, [:Yes, :No, :Yes, :No])
        @test issetequal(exploded_nt2.Sprinkler, [:On, :Off, :On, :Off])

        exploded_nt3 = EnhancedBayesianNetworks._explode_missing_single_line(nt2, states_space2)
        @test nrow(exploded_nt3) == 6
        @test issetequal(exploded_nt3.Rain, [:Yes, :No, :Pooring, :Yes, :No, :Pooring])
    end

    @testset "Expectation Step" begin
        net1 = deepcopy(bn)
        net1 = learn_parameters_MLE(df, net1)
        nt1 = copy(df_m[2, :])
        exploded_nt1 = EnhancedBayesianNetworks._expectation_step_single_missing_line(nt1, net1, states_space)
        @test nrow(exploded_nt1) == 2
        @test isapprox(exploded_nt1.weight, [0.89838, 0.10161], atol=0.001)

        nt2 = copy(df_m[3, :])
        exploded_nt2 = EnhancedBayesianNetworks._expectation_step_single_missing_line(nt2, net1, states_space)
        @test nrow(exploded_nt2) == 4
        @test isapprox(exploded_nt2.weight, [0.185251, 0.143808, 0.670940, 0.0], atol=0.001)

        @test_throws ErrorException("node Rain has a defined scenario state Pooring that is not among its possible states [:No, :Yes]") EnhancedBayesianNetworks._expectation_step_single_missing_line(nt2, net1, states_space2)

        exploded_nt2 = EnhancedBayesianNetworks._expectation_step_single_missing_line(nt2, net1, states_space3)
        @test nrow(exploded_nt2) == 4
        @test isapprox(exploded_nt2.weight, [0.185251, 0.143808, 0.670940, 0.0], atol=0.001)
    end

    @testset "BN from incomplete Dataframe" begin
        net1 = deepcopy(bn)
        e_bn1 = @suppress EnhancedBayesianNetworks._bn_from_incomplete_df(df_m, net1, 100, states_space)
        @test e_bn1.A == bn.A
        @test isapprox(e_bn1.nodes[1].cpt.data.Π, [0.5229331, 0.47706687], atol=0.01)
        @test isapprox(e_bn1.nodes[2].cpt.data.Π, [0.8974266, 0.102573, 0.409045, 0.5909], atol=0.01)
        @test isapprox(e_bn1.nodes[3].cpt.data.Π, [0.097108, 0.902891, 0.7354979, 0.26450], atol=0.01)
        @test isapprox(e_bn1.nodes[4].cpt.data.Π, [1.0, 0.0, 0.09728167, 0.902718, 0.06960197, 0.9303980, 0.0082798605, 0.9917201], atol=0.01)

        net2 = deepcopy(bn)
        @test_throws ErrorException("provided dataframe does not contain the following states for the following nodes Dict{Any, Any}(:Rain => [:Pooring])") EnhancedBayesianNetworks._bn_from_incomplete_df(df_m, net2, 100, states_space2)

        net3 = deepcopy(bn)
        e_bn3 = @suppress EnhancedBayesianNetworks._bn_from_incomplete_df(df_m, net3, 100, states_space3)
        @test e_bn3.A == bn.A
        @test isapprox(e_bn3.nodes[1].cpt.data.Π, [0.5229331, 0.47706687], atol=0.01)
        @test isapprox(e_bn3.nodes[2].cpt.data.Π, [0.8974266, 0.102573, 0.409045, 0.5909], atol=0.01)
        @test isapprox(e_bn3.nodes[3].cpt.data.Π, [0.097108, 0.902891, 0.7354979, 0.26450], atol=0.01)
        @test isapprox(e_bn3.nodes[4].cpt.data.Π, [1.0, 0.0, 0.0, 0.097281, 0.0, 0.9027, 0.0696019, 0.0, 0.9303, 0.008279, 0.0, 0.99172], atol=0.01)
    end

    @testset "Learn parameters EM" begin
        net1 = deepcopy(bn)
        net2 = deepcopy(bn)
        net3 = deepcopy(bn)

        MLE_bn1 = deepcopy(bn)
        MLE_bn1 = learn_parameters_MLE(df, MLE_bn1)
        e_bn1 = @suppress learn_parameters_EM(df_m, net1, 100, states_space)
        @test e_bn1.A == MLE_bn1.A
        @test e_bn1.topology == MLE_bn1.topology
        @test isapprox(e_bn1.nodes[1].cpt.data.Π, MLE_bn1.nodes[1].cpt.data.Π, atol=0.1)
        @test isapprox(e_bn1.nodes[2].cpt.data.Π, MLE_bn1.nodes[2].cpt.data.Π, atol=0.1)
        @test isapprox(e_bn1.nodes[3].cpt.data.Π, MLE_bn1.nodes[3].cpt.data.Π, atol=0.2)
        @test isapprox(e_bn1.nodes[4].cpt.data.Π, MLE_bn1.nodes[4].cpt.data.Π, atol=0.2)

        MLE_bn2 = deepcopy(bn)
        MLE_bn2 = learn_parameters_MLE(df, MLE_bn2, states_space3)
        e_bn2 = @suppress learn_parameters_EM(df_m, net2, 100, states_space3)
        @test isapprox(e_bn2.nodes[1].cpt.data.Π, MLE_bn2.nodes[1].cpt.data.Π, atol=0.1)
        @test isapprox(e_bn2.nodes[2].cpt.data.Π, MLE_bn2.nodes[2].cpt.data.Π, atol=0.1)
        @test isapprox(e_bn2.nodes[3].cpt.data.Π, MLE_bn2.nodes[3].cpt.data.Π, atol=0.2)
        @test isapprox(e_bn2.nodes[4].cpt.data.Π, MLE_bn2.nodes[4].cpt.data.Π, atol=0.2)

        @test_throws ErrorException("provided dataframe does not contain the following states for the following nodes Dict{Any, Any}(:Rain => [:Pooring])") learn_parameters_EM(df_m, net3, 100, states_space2)
    end
end
