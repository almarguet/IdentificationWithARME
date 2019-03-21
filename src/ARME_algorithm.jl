""" Reduction of the result of the S-step: gives the acceptation rate and update the population parameters. """
function reduction(res::Array{Tuple{Array{Float64,1},Population{T}},1}, meta_pop::Metapopulation, meta_pop_prop::Metapopulation) where T<:Union{Float64, Array{Float64,2}}
    acc = zeros(4)
    for i = 1:length(res)
        acc .+= res[i][1]
        copy_some_parameters!(meta_pop.subpopulations[i],res[i][2])
        copy_some_parameters!(meta_pop_prop.subpopulations[i],meta_pop.subpopulations[i])
    end
    return acc
end

function reduction(res::Array{Population{T},1}, meta_pop::Metapopulation) where T<:Union{Float64, Array{Float64,2}}
    for i = 1:length(res)
        copy_some_parameters!(meta_pop.subpopulations[i],res[i])
    end
end
""" Computation of the Fisher information matrix for nondiagonal global covariance. """
function compute_FI(OmegaT::Type{Array{Float64,2}}, meta_pop, meta_data, Delta, gamma, grad, hess, G, h; noheritability=false)
    if noheritability
        g! = (grad,y)-> ForwardDiff.gradient!(grad,x->global_likelihood(OmegaT, meta_pop,meta_data,meta_pop.heritability,
            [x[1],x[2],x[3]],[x[4] 0. 0.; x[5] x[6] 0. ; x[7] x[8] x[9]],x[10]),y)
        h! = (hess,y)-> ForwardDiff.hessian!(hess,x->global_likelihood(OmegaT, meta_pop,meta_data,meta_pop.heritability,
            [x[1],x[2],x[3]],[x[4] 0. 0.; x[5] x[6] 0. ; x[7] x[8] x[9]],x[10]),y)
        g!(grad,[meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h])
        h!(hess,[meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h])
    else
        g! = (grad,y)-> ForwardDiff.gradient!(grad,x->global_likelihood(OmegaT, meta_pop,meta_data, Matrix(Diagonal([x[1], x[2],x[3]])),
            [x[4],x[5],x[6]], [x[7] 0. 0. ; x[8] x[9] 0. ; x[10] x[11] x[12]], x[13]),y)
        h! = (hess,y)-> ForwardDiff.hessian!(hess,x->global_likelihood(OmegaT, meta_pop,meta_data,Matrix(Diagonal([x[1], x[2],x[3]])),
            [x[4],x[5],x[6]], [x[7] 0. 0. ; x[8] x[9] 0. ; x[10] x[11] x[12]], x[13]),y)
        g!(grad,[meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
         meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h])
        h!(hess,[meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
         meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h])
    end
    Delta = Delta + gamma*(grad-Delta)
    G = G + gamma*(hess + grad*grad'-G)
    H = G - Delta*Delta'
    return H, Delta, G
end

""" Computation of the Fisher information matrix for diagonal global covariance. """
function compute_FI(OmegaT::Type{Array{Float64,1}}, meta_pop, meta_data, Delta, gamma, grad, hess, G, h; noheritability=false)
    if noheritability
        g! = (grad,y)-> ForwardDiff.gradient!(grad,x->global_likelihood(OmegaT, meta_pop, meta_data, meta_pop.heritability,
            [x[1],x[2],x[3]],Matrix(Diagonal([x[4],x[5],x[6]])),x[7]), y)
        h! = (hess,y)-> ForwardDiff.hessian!(hess,x->global_likelihood(OmegaT, meta_pop,meta_data,meta_pop.heritability,
            [x[1],x[2],x[3]],Matrix(Diagonal([x[4],x[5],x[6]])),x[7]), y)
        g!(grad,[meta_pop.global_mean[1],meta_pop.global_mean[2],meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1],meta_pop.global_covariance[2,2],meta_pop.global_covariance[3,3],h])
        h!(hess,[meta_pop.global_mean[1],meta_pop.global_mean[2],meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1],meta_pop.global_covariance[2,2],meta_pop.global_covariance[3,3],h])
    else
        g! = (grad,y)-> ForwardDiff.gradient!(grad,x->global_likelihood(OmegaT, meta_pop, meta_data, Matrix(Diagonal([x[1], x[2],x[3]])),
            [x[4],x[5],x[6]], Matrix(Diagonal([x[7],x[8],x[9]])), x[10]), y)
        h! = (hess,y)-> ForwardDiff.hessian!(hess,x->global_likelihood(OmegaT, meta_pop, meta_data, Matrix(Diagonal([x[1], x[2],x[3]])),
            [x[4],x[5],x[6]], Matrix(Diagonal([x[7],x[8],x[9]])), x[10]), y)
        g!(grad, [meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
            meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1], meta_pop.global_covariance[2,2], meta_pop.global_covariance[3,3], h])
        h!(hess, [meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
            meta_pop.global_mean[1],meta_pop.global_mean[2],meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1],meta_pop.global_covariance[2,2],meta_pop.global_covariance[3,3],h])
    end
    Delta = Delta + gamma*(grad-Delta)
    G = G + gamma*(hess + grad*grad'-G)
    H = G - Delta*Delta'
    return H, Delta, G
end
""" Number of parameters to estimate, depending on their type. """
number_of_param(OmegaT::Type{Array{Float64,2}}; noheritability=false) = 13-3*noheritability
number_of_param(OmegaT::Type{Array{Float64,1}}; noheritability=false) = 10-3*noheritability

"""
# Arguments
-`OmegaT`: type of omega (Type{Array{Float64,1}} for the diagonal case or Type{Array{Float64,2}})

-`meta_data`: data for the estimations.

-`meta_pop`: initial population for the estimations.

-`meta_pop_prop`: initial population for the proposals in the MCMC procedure.

-`t_osmo`: vector of osmotic shocks.

-`dt`: step size used for the computation of protein density.

-`km`: fixed value of mRNA production rate.

-`h`: initial noise of measure.

-`N_burnè: number of burn-in iterations.

-`N_MCMC`: number of iterations of the total S-step with all kernels.

-`nMC1, nMC2, nMC3`: number of iterations with the each kernel in the MCMC procedure.

-`N`: number of ARME iterations.

-`N_begin_mem`: number of iterations without memory for the estimation.

-`sig`: initial values of the standard deviation for the random walk in the MCMC procedure with the third kernel.

-`delta`: parameter for the adaptative change of sig. Usually set to 0.4.

# Keyword Arguments

-`noheritability`: Delfault=false. If true, computation without heritability (A=0).

"""
function ARME(OmegaT, meta_data, meta_pop, meta_pop_prop, t_osmo, dt, km, h, N_burn, N_MCMC,
        nMC1, nMC2, nMC3, N, N_begin_memory, sig, delta; noheritability=false)
    m = length(meta_pop.global_mean)
    # Initialization of sufficient statistics
    s0 = 0.
    s1 = zeros(m,m)
    s2 = zeros(m,m)
    s3 = zeros(m,m)
    s4 = zeros(m)
    s5 = zeros(m)
    s6 = zeros(m,m)
    s7 = zeros(m)
    # Bounds for the maximization step
    lower, upper = bound_for_maximization(meta_pop.heritability, OmegaT, noheritability)
    gamma = function_for_stabilization(N, N_begin_memory)
    # Initial values of population parameters
    heritability = [copy(meta_pop.heritability)]
    global_mean = [copy(meta_pop.global_mean)]
    global_covariance =[copy(meta_pop.global_covariance)]
    h_sequence = [h]
    FI = Array{Array{Float64,2},1}()
    loglik = Array{Float64,1}()
    n_param = number_of_param(OmegaT ; noheritability=noheritability)
    Delta = zeros(n_param)
    G = zeros(n_param,n_param)
    H = zeros(n_param,n_param)
    grad = zeros(n_param)
    hess = zeros(n_param,n_param)
    gene = 0
    # Burn-in iterations
    for i = 1:N_burn
        res = @distributed append! for k = 1:meta_pop.n_subpopulations
            println(string("burn-in iteration ",i," for subpopulation ",k))
            acc = zeros(4)
            simulation_step_MCMC_1(meta_pop.subpopulations[k],meta_pop_prop.subpopulations[k],nMC1,meta_data[k],t_osmo, dt, km, h)
            # Synchronization of the values of pop and pop_prop
            copy_some_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k])
            for v = 1:meta_pop.subpopulations[k].size
                simulation_step_MCMC_2(meta_pop.subpopulations[k],meta_pop_prop.subpopulations[k],nMC2,meta_data[k],t_osmo, dt, km, v, h)
                # Synchronization of the values of pop and pop_prop
                copy_some_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k]; u=v)
                acc[4] += simulation_step_MCMC_3_all(meta_pop.subpopulations[k], meta_pop_prop.subpopulations[k],
                    nMC3,meta_data[k],t_osmo, dt, km, sig[4], v, h)
                for k2 = 1:3
                    acc[k2] += simulation_step_MCMC_3_one_param(meta_pop.subpopulations[k],
                        meta_pop_prop.subpopulations[k], nMC3, meta_data[k], t_osmo, dt, km, sig[k2], v, h, k2)
                    # Synchronization of the values of pop and pop_prop
                    copy_some_parameters!(meta_pop_prop.subpopulations[k],meta_pop.subpopulations[k]; u=v)
                end
            end
            [(acc, meta_pop.subpopulations[k])]
        end
        acc = reduction(res, meta_pop, meta_pop_prop)
        println(string("acceptation rate after burn-in = ", acc ./ (2*nMC3*meta_pop.total_size)))
        # update of the standard deviation of the random walk to obtain an acceptance rate equal to 0.3
        sig = sig.*(1 .+ delta.*(acc./(2*nMC3*meta_pop.total_size) .- 0.3))
    end
    # ARME iterations
    for k = 1:N
        println(string("iteration ",k," of ARME"))
        # S-step
        res = @distributed append! for i= 1:meta_pop.n_subpopulations
            acc = zeros(4)
            for i_MCMC = 1:N_MCMC
                simulation_step_MCMC_1(meta_pop.subpopulations[i],meta_pop_prop.subpopulations[i],nMC1,meta_data[i],t_osmo, dt, km, h)
                # Synchronization of the values of pop and pop_prop
                copy_some_parameters!(meta_pop_prop.subpopulations[i],meta_pop.subpopulations[i])
                for v= 1:meta_pop.subpopulations[i].size
                    simulation_step_MCMC_2(meta_pop.subpopulations[i],meta_pop_prop.subpopulations[i],nMC2,meta_data[i],t_osmo, dt, km, v, h)
                    # Synchronization of the values of pop and pop_prop
                    copy_some_parameters!(meta_pop_prop.subpopulations[i],meta_pop.subpopulations[i]; u=v)
                    acc[4] += simulation_step_MCMC_3_all(meta_pop.subpopulations[i], meta_pop_prop.subpopulations[i],
                        nMC3,meta_data[i], t_osmo, dt, km, sig[4], v, h)
                    for k2 = 1:3
                        acc[k2] += simulation_step_MCMC_3_one_param(meta_pop.subpopulations[i],
                            meta_pop_prop.subpopulations[i], nMC3, meta_data[i], t_osmo, dt, km, sig[k2], v, h, k2)
                        # Synchronization of the values of pop and pop_prop
                        copy_some_parameters!(meta_pop_prop.subpopulations[i],meta_pop.subpopulations[i]; u=v)
                    end
                end
            end
            [(acc, meta_pop.subpopulations[i])]
        end
        acc = reduction(res, meta_pop, meta_pop_prop)
        sig = sig.*(1 .+ delta.*(acc./(2*N_MCMC*nMC3*meta_pop.total_size) .- 0.3))
        # Update of sufficent statistics
        s0,s1,s2,s3,s4,s5,s6,s7 = sufficient_stat(meta_pop, meta_data, s0, s1, s2, s3, s4, s5, s6, s7, gamma[k])
        # M-step
        h = maximization_step_optim(OmegaT, meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,lower,upper,h; noheritability=noheritability)
        # Computation of the Fisher Information Matrix
        H, Delta, G = compute_FI(OmegaT, meta_pop, meta_data, Delta, gamma[k], grad, hess, G, h; noheritability=noheritability)
        # Update of population parameters of meta_pop_prop
        copy_global_parameters!(meta_pop_prop, meta_pop)
        # Computation of the current value of the loglikelihood
        push!(loglik,global_likelihood(OmegaT,
            meta_pop,meta_data,meta_pop.heritability,meta_pop.global_mean,meta_pop.global_covariance,h))
        push!(FI,copy(H))
        push!(h_sequence,h)
        println(string("h = ", h))
        if !noheritability
            push!(heritability,copy(meta_pop.heritability))
            println(string("A = ",meta_pop.heritability))
        end
        push!(global_mean,copy(meta_pop.global_mean))
        println(string("b = ",meta_pop.global_mean))
        push!(global_covariance,copy(meta_pop.global_covariance))
        println(string("Omega = ",meta_pop.global_covariance))
    end
    if noheritability
        return global_mean, global_covariance, h_sequence, FI, loglik
    else
        return heritability, global_mean, global_covariance, h_sequence, FI, loglik
    end
end
