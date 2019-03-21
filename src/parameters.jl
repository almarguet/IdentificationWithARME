""" Compute the array of ancestor indices in the case of several subpopulations of the same type, complete tree or branch. Default: complete tree. """
function create_ancestor_index(n_gene; branch = false)
    ancestor_index_meta = Array{Array{Int64,1},1}()
    if branch
        for i = 1:length(n_gene)
            ancestor_index = [vcat(0,1:n_gene[i]-1)]
            push!(ancestor_index_meta, ancestor_index)
        end
    else
        for i = 1:length(n_gene)
            ancestor_index = zeros(Int,2^n_gene[i]-1)
            for j = 1:2^(n_gene[i]-1)-1
                ancestor_index[2*j:2*j+1] = j*ones(Int,2)
            end
            push!(ancestor_index_meta, ancestor_index)
        end
    end
    return ancestor_index_meta
end

""" Create an array of birth and death times for a fixed lifetime. """
function create_birth_death_time_meta(lifetime, ancestor_index_meta)
    birth_death_time_meta =  []
    n_subpopulations = length(ancestor_index_meta)
    for i = 1:n_subpopulations
        current_pop_size = length(ancestor_index_meta[i])
        birth_death_time = zeros(2,current_pop_size)
        birth_death_time[:,1] = [0,lifetime]
        for k = 2:current_pop_size
            birth_death_time[:,k] =[birth_death_time[2,ancestor_index_meta[i][k]],birth_death_time[2,ancestor_index_meta[i][k]]+lifetime]
        end
        push!(birth_death_time_meta,birth_death_time)
    end
    return convert(Array{Array{Float64,2},1},birth_death_time_meta)
end

"""
# Arguments

-`n_gene`: size of the population in number of generation.

-`lifetime`: lifetime of each individual.

# Keyword Arguments

-`branch`: default_value=false. If true, generate parameters for a branch, else for a complete tree.

# Return
-`n`: a vector giving the size of each subpopulation.

-`n_subpopulations`: the number of subpopulations.

-`total_size`: the sum of all subpopulations sizes.

-`ancestor_index_meta`: an array of arrays giving the index of the ancestor of each individual in each subpopulation.

-`birth_death_time_meta`: gives for each subpopulation the birth and death time of each individual.

"""
function parameters_pop(n_gene, lifetime; branch = false)
    n_subpopulations = length(n_gene)
    ancestor_index_meta = create_ancestor_index(n_gene; branch = branch)
    birth_death_time_meta = create_birth_death_time_meta(lifetime, ancestor_index_meta)
    n = [length(ancestor_index_meta[i]) for i = 1:n_subpopulations]
    total_size = sum(n)
    return n, n_subpopulations, total_size, ancestor_index_meta, birth_death_time_meta
end


"""
# Arguments

-`lifetime`: lifetime of each individual.

-`n_gene`: size of the population in number of generation.

-`waiting`: waiting time between two osmotic shocks.

-`duration`: duration of a shock.

# Return

-`t_osmo`: array of times of osmotic shocks. First column: beginning times, second column: end times.

-`n_osmo`: number of osmotic shocks.

-`tspan`: couple of beginning and end times of the population.

"""
function create_t_osmo(lifetime, n_gene, waiting, duration)
    tspan = (0.,lifetime*maximum(n_gene))
    n_osmo = Int(round(tspan[2]/waiting))
    t_osmo = zeros(n_osmo,2)
    for i = 1:n_osmo
        t_osmo[i,1] = waiting*i
        t_osmo[i,2] = waiting*i+duration
    end
    return t_osmo, n_osmo, tspan
end

"""
# Arguments

-`tspan`: couple of beginning and end times of the population.

-`tstep`: length of time steps.

# Return

-`n_time`: the number of timesteps.

-`v_time`: the vector of timesteps.
"""
function create_time_steps(tspan, tstep )
    n_time = Int(round((tspan[2]-tspan[1])/tstep+1))
    v_time = range(tspan[1],stop = tspan[2],length = n_time)
    return n_time, v_time
end

function function_for_stabilization(N,N_begin)
    return vcat(ones(N_begin),[1/(k-N_begin) for k=(N_begin +1):N])
end

""" Return the bounds for the maximization using the Interior Point Newton algorithm. """
function bound_for_maximization(a, OmegaT, noheritability)
    if OmegaT == Array{Float64,2} && !noheritability
        lower = [0.,0.,0.,-Inf, -Inf, -Inf, 0., -Inf, 0., -Inf, -Inf, 0., 0.]
        upper = [0.99,0.99,0.99, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]
    elseif OmegaT == Array{Float64,2}
        lower = [-Inf, -Inf, -Inf, 0., -Inf, 0., -Inf, -Inf, 0., 0.]
        upper = [ Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf, Inf]
    else
        lower = [0.,0.,0.,-Inf, -Inf, -Inf, 0., 0., 0., 0.]
        upper = [1.,1.,1., Inf, Inf, Inf, Inf, Inf, Inf, Inf]
    end
    return lower, upper
end
