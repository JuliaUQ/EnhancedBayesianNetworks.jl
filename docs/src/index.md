---
layout: home

hero:
  name: EnhancedBayesianNetworks.jl
  text: Bayesian networks with continuous, functional, and imprecise nodes
  tagline: Build enhanced Bayesian networks, reduce them through structural reliability analysis, and query them with exact or credal inference.
  actions:
    - theme: brand
      text: Getting Started
      link: /manual/gettingstarted
    - theme: alt
      text: Introduction
      link: /manual/introduction
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl

features:
  - title: Enhanced Bayesian networks
    details: Mix discrete, continuous, and functional nodes in a single model, with precise or imprecise (interval / probability-box) quantities.
  - title: Reduction & reliability
    details: Reduce an enhanced network to a Bayesian or credal network, evaluating functional nodes as structural reliability problems through UncertaintyQuantification.jl.
  - title: Exact & credal inference
    details: Variable elimination on Bayesian networks and lower/upper bounds on credal networks, alongside parameter learning and network plotting.

authors:
  - name: Andrea Perin
    platform: github
    link: https://github.com/andreaperin
  - name: Jasper Behrensdorf
    platform: github
    link: https://github.com/FriesischScott
  - name: Matteo Broggi
    platform: github
    link: https://github.com/teobros
  - name: Laurenz Knipper
    platform: github
    link: https://github.com/sitoryu
---

EnhancedBayesianNetworks.jl extends the classical Bayesian-network formalism with the
continuous and functional nodes of structural reliability analysis, and with imprecision at
every level. New here? Start with the [Introduction](manual/introduction.md) for the concepts,
or jump to [Getting Started](manual/gettingstarted.md) to install the package and run your first
model.
