using Documenter
using DocumenterCitations
using EnhancedBayesianNetworks

bib=CitationBibliography(joinpath(@__DIR__, "citations.bib"))

makedocs(; modules=[EnhancedBayesianNetworks], plugins=[bib], authors="Andrea Perin & Jasper Behrensdorf", pages=["Home" => "index.md", "References" => "references.md"], sitename="EnhancedBayesianNetworks.jl", source="src", build="build", warnonly=true, format=Documenter.HTML(prettyurls=get(ENV, "CI", nothing) == "true"), draft=false)