""" Post-processing : compute N_sample samples of the individual parameters using a Metropolis-Hasting algorithm.
meta_data: data used of identification.
meta_pop: a metapopulation with individual parameters corresponding to the one obtained at the end of the ARME procedure.
meta_pop_prop: same as meta_pop.
h: estimated noise of measure.
N_sample: size of the sample of individual parameters produced.
nMC1, nMC2, nMC3: number of iterations with the each kernel in the MCMC procedure.
t_osmo: vector of osmotic shocks.
sig_init: initial values of the standard deviation simulation_step_MCMC_3_one_param.
sig_tot_init: initial value of the standard deviation for simulation_step_MCMC_3_all.
delta: parameter for the adaptative change of sig. Usually set to 0.4.
dt: step size used for the computation of protein density.
km: fixed value for the mRNA production rate.
"""
function individual_parameters_estimation(meta_data, meta_pop, meta_pop_prop,
        h, N_sample, nMC1, nMC2, nMC3, t_osmo, sig_init, sig_tot_init, delta, dt, km)
    sampl = @distributed append! for k = 1:meta_pop.n_subpopulations
        new_sampl = zeros(Float64, N, 3, meta_pop.subpopulations[k].size)
        sig = sig_init
        sig_tot = sig_tot_init
        for i::Int = 1:N_sample
            println(string("computation of sample ",i," for subpopulation ",k))
            acc = zeros(3)
            acc_tot = 0.
            simulation_step_MCMC_1(meta_pop.subpopulations[k], meta_pop_prop.subpopulations[k], nMC1, meta_data[k], t_osmo, dt, km, h)
            # Synchronization of the values of pop and pop_prop
            copy_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k])
            for v = 1:meta_pop.subpopulations[k].size
                simulation_step_MCMC_2(meta_pop.subpopulations[k], meta_pop_prop.subpopulations[k], nMC2,meta_data[k], t_osmo, dt, km, v, h)
                # Synchronization of the values of pop and pop_prop
                copy_some_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k]; u=v)
                acc_tot += simulation_step_MCMC_3_all(meta_pop.subpopulations[k], meta_pop_prop.subpopulations[k],
                    nMC3, meta_data[k], t_osmo, dt, km, sig_tot, v, h)
                for k2 = 1:3
                    acc[k2] += simulation_step_MCMC_3_one_param(meta_pop.subpopulations[k],
                        meta_pop_prop.subpopulations[k], nMC3, meta_data[k], t_osmo, dt, km, sig[k2], v, h, k2)
                    # Synchronization of the values of pop and pop_prop
                    copy_some_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k]; u=v)
                end
            end
            sig_tot = sig_tot*(1 + delta*(acc_tot/(2*nMC3*meta_pop.subpopulations[k].size) - 0.3))
            sig = sig.*(1 .+ delta.*(acc./(2*nMC3*meta_pop.subpopulations[k].size) .- 0.3))
            new_sampl[i,:,:]  = meta_pop.subpopulations[k].individual_parameters
        end
        [new_sampl]
    end
    sampl
end
