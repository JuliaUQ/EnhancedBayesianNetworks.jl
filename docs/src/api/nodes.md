# Nodes

## Index

```@index
Pages = ["nodes.md"]
```

## Types

```@docs
DiscreteNode
ContinuousNode
ContinuousFunctionalNode
DiscreteFunctionalNode
ExactDiscretization
ApproximatedDiscretization
```

## Methods

```@docs
states
scenarios
parents
isroot
isprecise
sample(node::DiscreteNode, evidence::Evidence)
```