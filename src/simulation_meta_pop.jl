""" Simulation of individual parameters of pop for the individuals in indices. If stationnary = true, computation with respect to the stationnary law. """
function simulation_parameters(pop::Population ; indices_parameters=1:3, indices_individuals=0, stationnary=false)
    if indices_individuals == 0
        indices_individuals = 1:pop.size
    end
    n = length(pop.individual_parameters[indices_parameters,1])
    for u in indices_individuals
        if u>1 && !stationnary
            pop.individual_parameters[indices_parameters,u] = pop.heritability*pop.individual_parameters[indices_parameters,pop.ancestor_index[u]] .+
            (I-pop.heritability)*pop.global_mean[indices_parameters] .+ pop.global_sqrt_covariance[indices_parameters,indices_parameters]*randn(n)
            pop.exp_individual_parameters[indices_parameters,u] = exp.(pop.individual_parameters[indices_parameters,u])
        else
            pop.individual_parameters[indices_parameters,u] = pop.global_mean[indices_parameters] .+ pop.stationary_sqrt_covariance[indices_parameters,indices_parameters]*randn(n)
            pop.exp_individual_parameters[indices_parameters,u] = exp.(pop.individual_parameters[indices_parameters,u])
        end
    end
end

"""Returns a simulated metapopulation.

# Arguments

-`n_subpopulations`: number of subpopulations.

-`n`: vector of size of subpopulations.

-`ancestor_index_meta`: vector where the jth component is a vector containing the indexof the ancestor of the ith individual in the ith component of the jth subpopulation.

-`a`: matrix of heritability.

-`b`: global mean.

-`omega`: global covariance.

-`n_time`: number of sampling times.

-`v_time`: vector of sampling times.

-`birth_death_time_meta`: vector where the jth component is a vector containing birthtimes (first line) and deathtimes (second line) ith individual of the jth subpopulation in the ith column.

-`m`: dimension of the individual parameters.

-`n_osmo`: number of osmotic shocks.

-`t_osmo`: beginning (first column) and end (second column) times of osmotic shocks.

-`tau`: delay for the maturation of proteins (fixed in constant_SAEM.jl).

-`dt`: step size used for the computation of protein densities.

-`km`: fixed value of parameters km for mRNA production.

# Keyword Arguments

-`index_time`: specify the indices in `v_time` where an individual is observed, default_value=0: each time between birth and death.

 """
function simulation_metapop(n_subpopulations, n, ancestor_index, a, b, omega, n_time, v_time,
    birth_death_time_meta, m, n_osmo, t_osmo, tau, dt, km ; index_time=0)
    meta_pop = Metapopulation(n_subpopulations, n, ancestor_index, a, b, omega, n_time, v_time,
        birth_death_time_meta, m, n_osmo, t_osmo, tau ; index_time=index_time)
    for i = 1:meta_pop.n_subpopulations
        pop = meta_pop.subpopulations[i]
        # Simulation of individual parameters
        simulation_parameters(meta_pop.subpopulations[i])
        # Computation of the dynamic for proteins
        protein_density_diff(pop, t_osmo, dt, km)
    end
    return meta_pop
end
