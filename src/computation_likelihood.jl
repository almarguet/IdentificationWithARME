""" Update of sufficient statistics."""
function sufficient_stat(meta_pop, meta_data, s0, s1, s2, s3, s4, s5, s6, s7, gamma::Float64)
    s0 = (1-gamma)*s0
    s1 = (1-gamma)*s1
    s2 = (1-gamma)*s2
    s3 = (1-gamma)*s3
    s4 = (1-gamma)*s4
    s5 = (1-gamma)*s5
    s6 = (1-gamma)*s6
    s7 = (1-gamma)*s7
    current_pop::Population = meta_pop.subpopulations[1]
    current_data::Array{Float64,2} = meta_data[1]
    for k::Int = 1:meta_pop.n_subpopulations
        current_pop = meta_pop.subpopulations[k]
        current_data = meta_data[k]
        s0 = s0 +gamma*sum((current_pop.protein_density[current_pop.individual_index_time[:,1],1]-current_data[current_pop.individual_index_time[:,1],1]).^2)
        for u = 2:current_pop.size
            s0 = s0 +gamma*sum((current_pop.protein_density[current_pop.individual_index_time[:,u],u]-current_data[current_pop.individual_index_time[:,u],u]).^2)
            s1[:,:] = s1[:,:]+gamma*current_pop.individual_parameters[:,u]*current_pop.individual_parameters[:,u]'
            s2[:,:] = s2[:,:]+gamma*current_pop.individual_parameters[:,u]*current_pop.individual_parameters[:,current_pop.ancestor_index[u]]'
            s3[:,:] = s3[:,:]+gamma*current_pop.individual_parameters[:,current_pop.ancestor_index[u]]*current_pop.individual_parameters[:,current_pop.ancestor_index[u]]'
            s5[:] = s5[:]+gamma*current_pop.individual_parameters[:,u]
            s4[:] = s4[:]+gamma*current_pop.individual_parameters[:,current_pop.ancestor_index[u]]
        end
        s6[:,:] = s6[:,:]+gamma*current_pop.individual_parameters[:,1]*current_pop.individual_parameters[:,1]'
        s7[:] = s7[:]+gamma*current_pop.individual_parameters[:,1]
    end
    return s0, s1, s2, s3, s4, s5, s6, s7
end

""" Computation of the likelihood using sufficient statistics in the case Omega nondiagonal. In this case, the optimization is based on the cholesky factorization of Omega. """
function likelihood_sufficient_stat(OmegaT::Type{Array{Float64,2}}, meta_pop,
        s0, s1, s2, s3, s4, s5, s6, s7, a, b, sqrt_omega, h)
    omega = sqrt_omega*sqrt_omega'
    sigma = zeros(eltype(sqrt_omega[1,1]),3,3)
    for i = 1:3
        for j = 1:3
            sigma[i,j] = omega[i,j]/(1-a[i,i]*a[j,j])
        end
    end
    sigma_inverse = sigma^(-1)
    w = omega^(-1)
    sampling_size = 0.
    for i = 1:meta_pop.n_subpopulations
      for u = 1:meta_pop.subpopulations[i].size
        sampling_size = sampling_size + length(findall(meta_pop.subpopulations[i].individual_index_time[:,u]))
      end
    end
    res = 2*sampling_size*log(h) + s0/h^2 + tr(s6*sigma_inverse) - tr(b*s7'*sigma_inverse)- tr(s7*b'*sigma_inverse) + meta_pop.n_subpopulations*tr(b*b'*sigma_inverse) +
    tr(s1*w) - tr(s2*a*w)- tr(s2'*w*a) - tr(b*s5'*w*(I-a)) -tr(s5*b'*(I-a)*w) + tr(b*b'*(I-a)*w*(I-a))*(meta_pop.total_size-meta_pop.n_subpopulations) +
    tr(s3*a*w*a) + tr(b*s4'*a*w*(I-a)) + tr(s4*b'*(I-a)*w*a) +
    (meta_pop.total_size-meta_pop.n_subpopulations)*log(prod(diag(sqrt_omega))^2) + meta_pop.n_subpopulations*log(det(sigma))
end


""" Computation of the likelihood using sufficient statistics in the case Omega diagonal. """
function likelihood_sufficient_stat(OmegaT::Type{Array{Float64,1}}, meta_pop,
        s0, s1, s2, s3, s4, s5, s6, s7, a, b, omega, h)
    sampling_size = 0.
    s1 = diag(s1)
    s2 = diag(s2)
    s3 = diag(s3)
    s6 = diag(s6)
    for i = 1:meta_pop.n_subpopulations
      for u = 1:meta_pop.subpopulations[i].size
        sampling_size = sampling_size + length(findall(meta_pop.subpopulations[i].individual_index_time[:,u]))
      end
    end
    res = 2*sampling_size*log(h)+s0/h^2
    for i = 1:3
    res = res + (s6[i]-2*b[i]*s7[i]+meta_pop.n_subpopulations*b[i]^2)/omega[i]*(1-a[i]^2)+
        (s1[i]-2*a[i]*s2[i]-2*(1-a[i])*b[i]*s5[i] + (1-a[i])^2*b[i]^2*(meta_pop.total_size-meta_pop.n_subpopulations) +
        a[i]^2*s3[i]+2*a[i]*(1-a[i])*b[i]*s4[i])/omega[i] +
        meta_pop.total_size*log(omega[i])-meta_pop.n_subpopulations*log(1-a[i]^2)
    end
    res
end

""" Computation of the likelihood (without sufficient statistics) in the case of a diagonal global covariance. """
function global_likelihood(OmegaT::Type{Array{Float64,1}}, meta_pop,meta_data,a,b,omega,h)
    omega_inverse = omega^(-1)
    # types must adapt in order to do use the automatic differentiation.
    sigma = zeros(eltype(omega[1,1]),3,3)
    for i = 1:3
        for j = 1:3
            sigma[i,j] = omega[i,j]/(1-a[i,i]*a[j,j])
        end
    end
    sigma_inverse = sigma^(-1)
    pop = meta_pop.subpopulations[1]
    res = (pop.individual_parameters[:,1]-b)'*sigma_inverse*(pop.individual_parameters[:,1]-b)
    for u = 2:pop.size
        res = res + (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))'*omega_inverse*
        (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))
    end
    index_current::Array{Int,1} = []
    for u = 1:pop.size
      index_current = findall(pop.individual_index_time[:,u])
      for i = 1:length(index_current)
        res = res + (meta_data[1][index_current[i],u]-pop.protein_density[index_current[i],u])^2/h^2
      end
    end
    for i = 2:meta_pop.n_subpopulations
        pop = meta_pop.subpopulations[i]
        res = res + (pop.individual_parameters[:,1]-b)'*sigma_inverse*(pop.individual_parameters[:,1]-b)
        for u = 2:pop.size
            res = res + (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))'*omega_inverse*
            (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))
        end
        for u = 1:pop.size
          index_current = findall(pop.individual_index_time[:,u])
          for l = 1:length(index_current)
            res = res + (meta_data[i][index_current[l],u]-pop.protein_density[index_current[l],u])^2/h^2
          end
        end
    end
    sampling_size = 0.
    for i = 1:meta_pop.n_subpopulations
      for u = 1:meta_pop.subpopulations[i].size
        sampling_size = sampling_size + length(findall(meta_pop.subpopulations[i].individual_index_time[:,u]))
      end
    end
    res = res/2 + (meta_pop.total_size- meta_pop.n_subpopulations)/2*log((2*pi)^3*det(omega)) + meta_pop.n_subpopulations/2*log((2*pi)^3*det(sigma)) + sampling_size/2*log(2*pi*h^2)
end

""" Computation of the likelihood (without sufficient statistics) in the case of a nondiagonal global covariance. """
function global_likelihood(OmegaT::Type{Array{Float64,2}}, meta_pop,meta_data,a,b,sqrt_omega,h)
    omega = sqrt_omega*sqrt_omega'
    omega_inverse = omega^(-1)
    sigma = zeros(eltype(omega[1,1]),3,3)
    for i = 1:3
        for j = 1:3
            sigma[i,j] = omega[i,j]/(1-a[i,i]*a[j,j])
        end
    end
    sigma_inverse = sigma^(-1)
    pop = meta_pop.subpopulations[1]
    res = (pop.individual_parameters[:,1]-b)'*sigma_inverse*(pop.individual_parameters[:,1]-b)
    for u = 2:pop.size
        res = res + (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))'*omega_inverse*
        (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))
    end
    index_current::Array{Int,1} = []
    for u = 1:pop.size
        index_current = findall(pop.individual_index_time[:,u])
        for i = 1:length(index_current)
        res = res + (meta_data[1][index_current[i],u]-pop.protein_density[index_current[i],u])^2/h^2
        end
    end
    for i = 2:meta_pop.n_subpopulations
        pop = meta_pop.subpopulations[i]
        res = res + (pop.individual_parameters[:,1]-b)'*sigma_inverse*(pop.individual_parameters[:,1]-b)
        for u = 2:pop.size
            res = res + (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))'*omega_inverse*
            (pop.individual_parameters[:,u]-(a*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-a)*b))
        end
        for u = 1:pop.size
            index_current = findall(pop.individual_index_time[:,u])
            for l = 1:length(index_current)
                res = res + (meta_data[i][index_current[l],u]-pop.protein_density[index_current[l],u])^2/h^2
            end
        end
    end
    sampling_size = 0.
    for i = 1:meta_pop.n_subpopulations
        for u = 1:meta_pop.subpopulations[i].size
            sampling_size = sampling_size + length(findall(meta_pop.subpopulations[i].individual_index_time[:,u]))
        end
    end
    res = res/2 + (meta_pop.total_size- meta_pop.n_subpopulations)/2*log((2*pi)^3*prod(diag(sqrt_omega))^2) +
        meta_pop.n_subpopulations/2*log((2*pi)^3*det(sigma)) + sampling_size/2*log(2*pi*h^2)
end

""" Proportional to p(y|phi,theta). """
function loglikelihood_y(data,pop::Population,h)
  res::Float64 = 0.
  index_current::Array{Int,1} = []
  for u = 1:pop.size
      index_current = findall(pop.individual_index_time[:,u])
      for i = 1:length(index_current)
          res = res + (data[index_current[i],u]-pop.protein_density[index_current[i],u]).^2
      end
  end
    return -1/2/h^2*res
end

""" Computation of the factors of the loglikelihood that changes when the parameters of u change,
     whitout the term corresponding to the draw of the parameters of u (because it corresponds to
     the proposal kernel 2 used in the Metropolis-Hastings procedure, so that it cancels in the ratio of the acceptance rate). """
function loglikelihood_changing_u(pop::Population,data,u,h)
    offspring_contribution::Float64 = 0.
    data_contribution::Float64 =0.
    centered_value_child::Array{Float64,1} = zero(pop.individual_parameters[:,1])
    current_index::Array{Int,1} = []
    for v in findall(pop.ancestor_index.== u) # choice of the parameters for the children of u
        centered_value_child = pop.individual_parameters[:,v]-(pop.heritability*pop.individual_parameters[:,u]+(I-pop.heritability)*pop.global_mean)
        offspring_contribution = offspring_contribution + centered_value_child'*pop.global_inverse*centered_value_child
    end
    for v in pop.offspring[u]
      current_index = findall(pop.individual_index_time[:,v])
      for j in current_index
        data_contribution = data_contribution + (data[j,v]-pop.protein_density[j,v])^2
      end
    end
    return -1/2*(offspring_contribution + # choice of the parameter for the children of u
        data_contribution/h^2) # because of the dependency of the protein density to phi and of the initial value for the protein density of the descendants
end

""" Computation of the loglikelihood terms where u appears. """
function individual_loglikelihood(pop::Population,data,u,h)
  data_contribution::Float64 = 0.
  offspring_contribution::Float64 = 0.
  current_index::Array{Int,1} = []
  centered_value_child::Array{Float64,1} = zero(pop.individual_parameters[:,1])
  for v in pop.offspring[u]
    current_index = findall(pop.individual_index_time[:,v])
    for j in current_index
      data_contribution = data_contribution +(data[j,v]-pop.protein_density[j,v])^2
    end
  end
  offspring::Array{Int,1} = findall(pop.ancestor_index.== u)
    if u>1
        centered_value = pop.individual_parameters[:,u]-(pop.heritability*pop.individual_parameters[:,pop.ancestor_index[u]]+(I-pop.heritability)*pop.global_mean)
        for v in offspring # choice of the parameters for the children of u
            centered_value_child = pop.individual_parameters[:,v]-(pop.heritability*pop.individual_parameters[:,u]+(I-pop.heritability)*pop.global_mean)
            offspring_contribution = offspring_contribution + centered_value_child'*pop.global_inverse*centered_value_child
        end
        return  -1/2*(centered_value'*pop.global_inverse*centered_value + # choice of the parameter for u
                offspring_contribution +  # choice of the parameters for the children of u
                data_contribution/h^2) # because of the dependency of the protein density to phi and of the initial value for the protein density of the descendants
    else #u = 1
        centered_value = (pop.individual_parameters[:,u]-pop.global_mean)
        for v in offspring # choice of the parameters for the children of u
            centered_value_child = pop.individual_parameters[:,v]-(pop.heritability*pop.individual_parameters[:,u]+(I-pop.heritability)*pop.global_mean)
            offspring_contribution = offspring_contribution + centered_value_child'*pop.global_inverse*centered_value_child
        end
        return -1/2*(centered_value'*pop.stationary_inverse*centered_value + # choice of the parameter for u
            offspring_contribution +
            data_contribution/h^2) # because of the dependency of the protein density to phi and of the initial value for the protein density of the descendants
    end
end
