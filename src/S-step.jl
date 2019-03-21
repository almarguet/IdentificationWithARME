""" Metropolis-Hasting algorithm with kernel q1. """
function simulation_step_MCMC_1(pop::Population, pop_prop::Population, nMC, data, t_osmo, dt, km, h)
    vrais0= loglikelihood_y(data, pop, h)
    vrais_prop::Float64 = 0.
    alpha::Float64  = 0.
    accepted::Float64 = 0
    for i = 1:nMC
        simulation_parameters(pop_prop)
        for u = 1:pop.size
            protein_density_diff(pop_prop, t_osmo, dt, km; individual_index=[u])
        end
        vrais_prop = loglikelihood_y(data, pop_prop, h)
        alpha= exp(min(0, vrais_prop-vrais0))
        if rand()<alpha
            accepted = accepted + 1
            swap_parameters!(pop, pop_prop)
            vrais0 = vrais_prop
        end
    end
end

""" Metropolis-Hasting algorithm with kernel q2, changing only the parameters of u
    (and by consequence the protein density of its descendants). """
function simulation_step_MCMC_2(pop::Population, pop_prop::Population, nMC, data, t_osmo, dt, km, u, h)
    vrais_prop::Float64 = 0.
    alpha::Float64  = 0.
    vrais0::Float64 = loglikelihood_changing_u(pop, data, u, h)
    accepted::Int64 = 0
    for i = 1:nMC
        simulation_parameters(pop_prop; indices_individuals = [u])
        for v in pop.offspring[u]
            protein_density_diff(pop_prop, t_osmo, dt, km; individual_index=[v])
        end
        vrais_prop = loglikelihood_changing_u(pop_prop, data, u, h)
        alpha = exp(min(0, vrais_prop-vrais0))
        if rand()<alpha
            accepted = accepted+1
            copy_some_parameters!(pop, pop_prop; u=u)
            vrais0 = vrais_prop
        end
    end
end

""" Metropolis-Hasting algorithm with kernel q3, changing only the parameters of u
    (and by consequence the protein density of its descendants). """
function simulation_step_MCMC_3_all(pop::Population, pop_prop::Population, nMC, data, t_osmo, dt, km, sigma, u, h)
    vrais_prop::Float64 = 0.
    alpha::Float64  = 0.
    vrais0::Float64 = individual_loglikelihood(pop, data, u, h)
    accepted::Int64 = 0
    copy_some_parameters!(pop_prop, pop; u=u)
    for i = 1:nMC
        pop_prop.individual_parameters[:,u] = pop.individual_parameters[:,u]+ sigma*randn(3)
        pop_prop.exp_individual_parameters[:,u] = exp.(pop_prop.individual_parameters[:,u])
        for v in pop.offspring[u]
            protein_density_diff(pop_prop, t_osmo, dt, km; individual_index=[v])
        end

        vrais_prop = individual_loglikelihood(pop_prop,data,u,h)
        alpha = exp(min(0,vrais_prop-vrais0))
        if rand()<alpha
            accepted = accepted +1
            copy_some_parameters!(pop,pop_prop; u=u)
            vrais0 = vrais_prop
        end
    end
    accepted
end

""" Metropolis-Hasting algorithm with kernel q3, updating the kth component of the parameter of u
    (and by consequence the protein density of its descendants). """
function simulation_step_MCMC_3_one_param(pop::Population, pop_prop::Population, nMC, data, t_osmo, dt, km, sigma, u, h, k)
    vrais_prop::Float64 = 0.
    alpha::Float64  = 0.
    vrais0::Float64 = individual_loglikelihood(pop, data, u, h)
    accepted::Int64 = 0
    copy_some_parameters!(pop_prop,pop; u=u)
    for i = 1:nMC
        pop_prop.individual_parameters[k,u] = pop.individual_parameters[k,u]+ sigma*randn()
        pop_prop.exp_individual_parameters[k,u] = exp(pop_prop.individual_parameters[k,u])
        for v in pop.offspring[u]
            protein_density_diff(pop_prop, t_osmo, dt, km; individual_index=[v])
        end
        vrais_prop = individual_loglikelihood(pop_prop, data, u, h)
        alpha = exp(min(0, vrais_prop-vrais0))
        if rand()<alpha
            accepted = accepted +1
            copy_some_parameters!(pop,pop_prop; u=u)
            vrais0 = vrais_prop
        end
    end
    accepted
end
