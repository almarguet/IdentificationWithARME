""" The value of the indicator function at time t of osmotic shocks given by t_osmo. """
function input(t::Float64,t_osmo::Array{Float64,2})
    x = findall(t.>=t_osmo[:,1])
    if (length(x)>0)&& (t <= t_osmo[x[end],2])
        return 1.
    else
        return 0.
  end
end

""" ODE function which corresponds to the ODE system"""
function fun!(du, u, param::Tuple{Array{Float64,2},Array{Float64,2}}, t)
    M = param[1]
    t_osmo = param[2]
    du[1] = M[1,1]*input(t,t_osmo) + M[1,2]*u[1]
    du[2] =  M[2,1]*u[1] + M[2,2]*u[2]
    nothing
end
""" Computation of the value of protein density of individual in individual_index. Default : computation for the entire population. """
function protein_density_diff(pop::Population, t_osmo::Array{Float64,2}, dt::Float64, km::Float64; individual_index=0)
    if individual_index  == 0
        individual_index = 1:pop.size
    end
    for i in individual_index
        x = vcat(pop.birth_death_time[1,i], pop.v_time[pop.individual_index_time[:,i]], pop.birth_death_time[2,i])
        tspan = (x[1],x[end])
        if !(x[1] == x[end])
            # Computation of the dynamic of protein concentration for individual i.
            u0 = [pop.mRNA_density_beginning[i], pop.protein_density_beginning[i]]
            prob = ODEProblem(fun!, u0, tspan,
                ([[km -pop.exp_individual_parameters[1,i]] ; [pop.exp_individual_parameters[2,i] -pop.exp_individual_parameters[3,i]]],
                t_osmo))
            sol = solve(prob; dt=dt)
            res = sol(x[2:end-1])
            pop.protein_density[pop.individual_index_time[:,i],i] = [res[k][2] for k=1:length(res)]
            # Computation of initial values for the offsprings.
            for v in findall(pop.ancestor_index.== i)
                pop.protein_density_beginning[v] = sol(pop.birth_death_time[1,v])[2]
                pop.mRNA_density_beginning[v] = sol(pop.birth_death_time[1,v])[1]
            end
        else
            pop.protein_density[pop.individual_index_time[:,i],i]  = zeros(sum(pop.individual_index_time[:,i]))
        end
    end
end
