""" Computation of differentials for nondiagonal covariance """
function computation_differentials(OmegaT::Type{Array{Float64,2}}, meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,h; noheritability=false)
    if noheritability
        g! = (G,y)-> ForwardDiff.gradient!(G,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,meta_pop.heritability,[x[1],x[2],x[3]],
            [x[4] 0. 0.; x[5] x[6] 0. ; x[7] x[8] x[9]], x[10]),y)
        h! = (H,y)-> ForwardDiff.hessian!(H,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,meta_pop.heritability,[x[1],x[2],x[3]],
            [x[4] 0. 0.; x[5] x[6] 0. ; x[7] x[8] x[9]], x[10]),y)
        x0 = [meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h]
        df = TwiceDifferentiable(x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,meta_pop.heritability,[x[1],x[2],x[3]],
            [x[4] 0. 0. ; x[5] x[6] 0. ; x[7] x[8] x[9]], x[10]), g! , h!, x0)
    else
        g! = (G,y)-> ForwardDiff.gradient!(G,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,Matrix(Diagonal([x[1],x[2],x[3]])), [x[4], x[5],x[6]],
            [x[7] 0. 0. ; x[8] x[9] 0. ; x[10] x[11] x[12]], x[13]),y)
        h! = (H,y)-> ForwardDiff.hessian!(H,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,Matrix(Diagonal([x[1],x[2],x[3]])), [x[4], x[5],x[6]],
            [x[7] 0. 0. ; x[8] x[9] 0. ; x[10] x[11] x[12]], x[13]),y)
        x0 = [meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
         meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
         meta_pop.global_sqrt_covariance[1,1],
         meta_pop.global_sqrt_covariance[2,1], meta_pop.global_sqrt_covariance[2,2],
         meta_pop.global_sqrt_covariance[3,1], meta_pop.global_sqrt_covariance[3,2], meta_pop.global_sqrt_covariance[3,3],h]
        df = TwiceDifferentiable(x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,Matrix(Diagonal([x[1],x[2],x[3]])), [x[4], x[5],x[6]],
            [x[7] 0. 0. ; x[8] x[9] 0. ; x[10] x[11] x[12]], x[13]), g! , h!, x0)
    end
    return df, x0
end

""" Computation of differentials for diagonal global covariance"""
function computation_differentials(OmegaT::Type{Array{Float64,1}}, meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,h; noheritability=false)
    if noheritability
        g! = (G,y)-> ForwardDiff.gradient!(G,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,diag(meta_pop.heritability),[x[1], x[2],x[3]] ,[x[4],x[5],x[6]], x[7]),y)
        h! = (H,y)-> ForwardDiff.hessian!(H,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,diag(meta_pop.heritability),[x[1], x[2],x[3]] ,[x[4],x[5],x[6]], x[7]),y)
        x0 = [meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1], meta_pop.global_covariance[2,2], meta_pop.global_covariance[3,3],h]
        df = TwiceDifferentiable(x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,diag(meta_pop.heritability),[x[1], x[2],x[3]] ,[x[4],x[5],x[6]], x[7]),g! , h!, x0)
    else
        g! = (G,y)-> ForwardDiff.gradient!(G,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,[x[1],x[2],x[3]],[x[4], x[5],x[6]] ,[x[7],x[8],x[9]], x[10]),y)
        h! = (H,y)-> ForwardDiff.hessian!(H,x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,[x[1],x[2],x[3]],[x[4], x[5],x[6]] ,[x[7],x[8],x[9]], x[10]),y)
        x0 = [meta_pop.heritability[1,1], meta_pop.heritability[2,2], meta_pop.heritability[3,3],
            meta_pop.global_mean[1], meta_pop.global_mean[2], meta_pop.global_mean[3],
            meta_pop.global_covariance[1,1], meta_pop.global_covariance[2,2], meta_pop.global_covariance[3,3],h]
        df = TwiceDifferentiable(x->likelihood_sufficient_stat(OmegaT,
            meta_pop,s0,s1,s2,s3,s4,s5,s6,s7,[x[1],x[2],x[3]],[x[4], x[5],x[6]] ,[x[7],x[8],x[9]], x[10]),g! , h!, x0)
    end
    return df, x0
end
""" Update meta_pop according to the results of the optimization algorithm given by res for diagonal global covariance. """
function result_optim(OmegaT::Type{Array{Float64,1}}, res, meta_pop; noheritability=false)
    if noheritability
        meta_pop.global_mean = copy([res[1],res[2],res[3]])
        meta_pop.global_covariance = Matrix(Diagonal(copy([res[4],res[5],res[6]])))
        meta_pop.global_sqrt_covariance = cholesky(meta_pop.global_covariance).U'
        return res[7]
    else
        for i = 1:3
            if res[i]<10^(-10)
                res[i] = 10^(-10)
            end
        end
        meta_pop.heritability = Matrix(Diagonal(copy([res[1],res[2],res[3]])))
        meta_pop.global_mean = copy([res[4],res[5],res[6]])
        meta_pop.global_covariance = Matrix(Diagonal(copy([res[7],res[8],res[9]])))
        meta_pop.global_sqrt_covariance = cholesky(meta_pop.global_covariance).U'
        return res[10]
    end
end
""" Update meta_pop according to the results of the optimization algorithm given by res for nondiagonal global covariance. """
function result_optim(OmegaT::Type{Array{Float64,2}}, res, meta_pop; noheritability=false)
    if noheritability
        meta_pop.global_mean = copy([res[1],res[2],res[3]])
        meta_pop.global_sqrt_covariance = [res[4] 0. 0. ; res[5] res[6] 0. ; res[7] res[8] res[9]]
        meta_pop.global_covariance = meta_pop.global_sqrt_covariance*meta_pop.global_sqrt_covariance'
        return res[10]
    else
        for i = 1:3
            if res[i]<10^(-10)
                res[i] = 10^(-10)
            elseif res[i] > 0.99 - 10^(-10)
                res[i] = 0.99 - 10^(-10)
            end
        end
        meta_pop.heritability = Matrix(Diagonal(copy([res[1],res[2],res[3]])))
        meta_pop.global_mean = copy([res[4],res[5],res[6]])
        meta_pop.global_sqrt_covariance = [res[7] 0. 0. ; res[8] res[9] 0. ; res[10] res[11] res[12]]
        meta_pop.global_covariance = meta_pop.global_sqrt_covariance*meta_pop.global_sqrt_covariance'
        return res[13]
    end
end
""" Maximization of the likelihood using Optim. """
function maximization_step_optim(OmegaT, meta_pop, s0, s1, s2, s3, s4, s5, s6, s7,
        lower, upper, h; noheritability=false)
    df, x0 = computation_differentials(OmegaT, meta_pop, s0, s1, s2, s3, s4, s5, s6, s7, h; noheritability=noheritability)
    dfc = TwiceDifferentiableConstraints(lower, upper)
    res = Optim.minimizer(optimize(df, dfc, x0, IPNewton(),Optim.Options(;allow_f_increases = true, successive_f_tol = 2, show_trace = false)))
    h = result_optim(OmegaT, res, meta_pop; noheritability=noheritability)
    if OmegaT == Array{Float64,1}
        meta_pop.stationary_covariance = stationnary_covariance_fun(meta_pop.heritability, meta_pop.global_covariance)
    end
    meta_pop.global_inverse = meta_pop.global_covariance^(-1)
    meta_pop.stationary_sqrt_covariance = cholesky(meta_pop.stationary_covariance).U'
    meta_pop.stationary_inverse = meta_pop.stationary_covariance^(-1)
    for i = 1:meta_pop.n_subpopulations
      meta_pop.subpopulations[i].heritability = meta_pop.heritability
      meta_pop.subpopulations[i].global_mean = meta_pop.global_mean[:]
      meta_pop.subpopulations[i].global_covariance = meta_pop.global_covariance[:,:]
      meta_pop.subpopulations[i].global_inverse = meta_pop.global_inverse[:,:]
      meta_pop.subpopulations[i].global_sqrt_covariance = meta_pop.global_sqrt_covariance[:,:]
      # meta_pop.subpopulations[i].stationary_mean = meta_pop.stationary_mean[:]
      meta_pop.subpopulations[i].stationary_covariance = meta_pop.stationary_covariance[:,:]
      meta_pop.subpopulations[i].stationary_inverse = meta_pop.stationary_inverse[:,:]
      meta_pop.subpopulations[i].stationary_sqrt_covariance = meta_pop.stationary_sqrt_covariance[:,:]
    end
    return h
end
