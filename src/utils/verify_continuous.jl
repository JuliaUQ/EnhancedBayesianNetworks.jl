function _verify_discretization(cpt::ContinuousConditionalProbabilityTable, discretization::AbstractDiscretization)
    if isroot(cpt) && isa(discretization, ApproximatedDiscretization)
        error("Root node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt")
    elseif !isroot(cpt) && isa(discretization, ExactDiscretization)
        error("Child node must have ApproximatedDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt")
    end
end