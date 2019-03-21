__precompile__(false)
module IdentificationWithARME
export parameters_pop, create_t_osmo, create_time_steps, simulate_data, simulation_metapop, ARME, load_estim_values, plot_with_data_quantile
using PyCall
# using Distributions
# using DataStructures
using LinearAlgebra
# using Roots
using Statistics
using Optim, NLSolversBase
using ForwardDiff
using DifferentialEquations
using JLD
using Distributed

@pyimport matplotlib.pyplot as plt
# @pyimport numpy
# @pyimport matplotlib.lines as lines
include("population.jl")
include("metapopulation.jl")
include("compute_protein_density.jl")
include("parameters.jl")
include("data_simulation.jl")
include("simulation_meta_pop.jl")
include("ARME_algorithm.jl")
include("M-step.jl")
include("computation_likelihood.jl")
include("S-step.jl")
include("individual_parameters_estimation.jl")
include("functions_for_plots.jl")

end # module
