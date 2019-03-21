using LinearAlgebra
using IdentificationWithARME
using JLD
include("constants.jl")

# Population parameters
m = 3 #number of individual parameters. Here, km=10(identifiablitity issues)
a_data = Matrix(0.5I,3,3)# a<1 to avoid blow up of parameters
b_data = [log(0.294), log(0.947), log(0.1)]
h_data = 20.
omega_data = [sqrt(0.1) 0. 0.; sqrt(0.05) sqrt(0.05) 0.;sqrt(0.1/3) sqrt(0.1/3) sqrt(0.1/3)]*
    [sqrt(0.1) 0. 0.; sqrt(0.05) sqrt(0.05) 0.;sqrt(0.1/3) sqrt(0.1/3) sqrt(0.1/3)]'


# Initial values of population parameters for the inference
a = Matrix(0.2I,3,3)
b = [-2.4, -0.1, -5]
h = 50.
omega = 2 .*omega_data

# Specify the type of omega:  OmegaT = Array{Float64,1} if omega is diagonal, else OmegaT = Array{Float64,2}
OmegaT = Array{Float64,2}

n_gene = [7] # Vector of number of generation for each subpopulation. Here, one subpopulation.
lifetime = 90. # Fixed value of lifetime for each individual.

# Computation of some parameters of the population, see parameters.jl
n, n_subpopulations, total_size, ancestor_index_meta, birth_death_time_meta = parameters_pop(n_gene, lifetime)

#initial values of mRNA and protein concentrations.
initial_densities = [0.,0.]

# Osmotic shocks
duration = 8.# Duration of osmotic chocs.
waiting = 30. # Time between 2 chocs.
# See parameters.jl
t_osmo, n_osmo, tspan = create_t_osmo(lifetime, n_gene, waiting, duration)

# Time steps
tstep = 1.
# See parameters.jl
n_time, v_time = create_time_steps(tspan, tstep)

# Parameters for the ARME algorithm
N = 2  # Number of ARME iterations.
N_MCMC = 2 # Number of repetitions of MCMC steps
N_burn = 2 # Number of burn-in iterations
nMC1 = 10 # Number of iterations with the first kernel (for the entire population)
nMC2 = 10 # Number of iterations with the second kernel (individual along a generation)
nMC3 = 10 # Number of iterations with the third kernel (random walk)
N_begin_memory = 80 # Number of iterations without memory for the estimation

# Step size for the resolution of the gene expression dynamical system
dt = 1e-02

# Initial values for the standard deviation of the random walk in the MCMC steps.
sigma = 0.1
sig = [0.1,0.1,0.1,0.1]
sig_tot = 0.1
delta = 0.4

# Data simulation or loading. Set load_data to true to used saved data. If load_data=false, new data are simulated.
load_data = true
file_data = "./data/SAEM_nondiagcov_diagA_h20_1.jld"
if load_data
    meta_data = load(file_data)["meta_data"]
else
    meta_data = simulate_data(n_subpopulations, n, ancestor_index_meta, a_data, b_data, omega_data, n_time, v_time,
        birth_death_time_meta, m, n_osmo, t_osmo, tau, dt, km, h_data; index_time=0)
end

# Simulation of a new population for the estimation
meta_pop = simulation_metapop(n_subpopulations, n, ancestor_index_meta, a, b, omega, n_time, v_time,
    birth_death_time_meta, m, n_osmo, t_osmo, tau, dt, km; index_time=0)

# Simulation of a new population for the proposal
meta_pop_prop = simulation_metapop(n_subpopulations, n, ancestor_index_meta, a, b, omega, n_time, v_time,
    birth_death_time_meta, m, n_osmo, t_osmo, tau, dt, km; index_time=0)

# Estimation procedure using ARME algorithm
heritability, global_mean,global_covariance, h_sequence, FI, loglik = ARME(OmegaT, meta_data,
meta_pop, meta_pop_prop, t_osmo, dt, km, h, N_burn, N_MCMC, nMC1, nMC2, nMC3, N, N_begin_memory, sig, delta)

# Saving the results
individual_parameters_estimated = []
individual_parameters_data = []
for i = 1:meta_pop_prop.n_subpopulations
  push!(individual_parameters_estimated, meta_pop.subpopulations[i].individual_parameters)
  if !load_data
      push!(individual_parameters_data, meta_pop_data.subpopulations[i].individual_parameters)
  end
end
file_name = "ARME_results.jld"
# If load_data, save the location of the data, else, save the data and individual parameters of the data.
if load_data
    save(file_name , "individual_parameters_estimated", individual_parameters_estimated,"file_data", file_data,
        "h", h_sequence, "heritability", heritability, "mean", global_mean, "omega", global_covariance, "loglik", loglik, "FI", FI)
else
    save(file_name , "individual_parameters_estimated", individual_parameters_estimated,
        "individual_parameters_data", individual_parameters_data, "meta_data", meta_data,
        "h", h_sequence, "heritability", heritability, "mean", global_mean, "omega", global_covariance, "loglik", loglik, "FI", FI)
end
