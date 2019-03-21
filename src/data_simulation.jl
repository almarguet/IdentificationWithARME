"""Returns simulated data.

# Arguments

-`n_subpopulations`: number of subpopulations.

-`n`: vector of size of subpopulations.

-`ancestor_index_meta`: vector where the jth component is a vector containing the indexof the ancestor of the ith individual in the ith component of the jth subpopulation.

-`a_data`: matrix of heritability.

-`b_data`: global mean.

-`omega_data`: global covariance.

-`n_time`: number of sampling times.

-`v_time`: vector of sampling times.

-`birth_death_time_meta`: vector where the jth component is a vector containing birthtimes (first line) and deathtimes (second line) ith individual of the jth subpopulation in the ith column.

-`m`: dimension of the individual parameters.

-`n_osmo`: number of osmotic shocks.

-`t_osmo`: beginning (first column) and end (second column) times of osmotic shocks.

-`tau`: delay for the maturation of proteins (fixed in constant_SAEM.jl).

-`dt`: step size used for the computation of protein densities.

-`h_data`: standard deviation of the measuring noise.

-`km`: fixed value of parameters km for mRNA production.

# Keyword Arguments

-`index_time`: specify the indices in `v_time` where an individual is observed, default_value=0: each time between birth and death.

 """
function simulate_data(n_subpopulations, n, ancestor_index_meta, a_data, b_data, omega_data, n_time, v_time,
    birth_death_time_meta, m, n_osmo, t_osmo, tau, dt, km, h_data; index_time=0)
    meta_data = Array{Array{Float64,2},1}()
    meta_pop_data= Metapopulation(n_subpopulations, n, ancestor_index_meta, a_data, b_data, omega_data, n_time, v_time,
        birth_death_time_meta, m, n_osmo, t_osmo, tau ; index_time=index_time)
    for i = 1:meta_pop_data.n_subpopulations
        pop = meta_pop_data.subpopulations[i]
        simulation_parameters(pop)
        protein_density_diff(pop, t_osmo, dt, km)
        bruit = h_data*rand(Normal(),meta_pop_data.n_time,meta_pop_data.subpopulations[i].size)
        push!(meta_data, meta_pop_data.subpopulations[i].protein_density + bruit)
    end
    return meta_data
end
