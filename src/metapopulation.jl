mutable struct Metapopulation{HeritabilityT}
    n_subpopulations::Int
    total_size::Int
    subpopulations::Array{Population,1}
    heritability::HeritabilityT
    global_mean::Array{Float64,1}
    global_covariance::Array{Float64,2}
    global_sqrt_covariance::Array{Float64,2}
    global_inverse::Array{Float64,2}
    stationary_covariance::Array{Float64,2}
    stationary_sqrt_covariance::Array{Float64,2}
    stationary_inverse::Array{Float64,2}
    n_time::Int
    v_time::Array{Float64,1}
    n_parameters::Int
    n_osmo_global::Int #  total number of osmotic shocks
    t_osmo_global::Array{Float64,2} # ith line : the ith osmotic shock, first column: beginning, second column: end
    tau::Float64 # delay for fluorescence
end
""" Creation of a metapopulation. If test_maximization = true, do not compute the offsring array. If index_time = 0 """
Metapopulation(n_subpopulations::Int64, sizes::Array{Int64,1},
        ancestor_index::Array{Array{Int64,1},1}, heritability::HeritabilityT,
        global_mean::Array{Float64,1}, global_covariance::Array{Float64,2},
        n_time::Int64, v_time::Union{AbstractRange,Array{Float64,1}},
        birth_death_time_meta::Array{Array{Float64,2},1}, m::Int64, n_osmo::Int64,
        t_osmo::Array{Float64, 2}, tau::Float64; test_maximization=false, index_time=0) where HeritabilityT<:Union{Float64, Array{Float64,2}} =
begin
    subpopulations = []
    for i = 1:n_subpopulations
        if index_time == 0
            push!(subpopulations, Population(sizes[i], ancestor_index[i], heritability,
                global_mean, global_covariance, n_time, v_time, birth_death_time_meta[i],
                m, n_osmo, t_osmo, tau; test_maximization = test_maximization, index_time=0))
        else
            push!(subpopulations, Population(sizes[i], ancestor_index[i], heritability,
                global_mean, global_covariance, n_time, v_time, birth_death_time_meta[i],
                m, n_osmo, t_osmo, tau; test_maximization = test_maximization, index_time=index_time[i]))
        end
    end
    total_size = sum(subpopulations[k].size for k =1:n_subpopulations)
    Metapopulation{typeof(heritability)}(n_subpopulations, total_size, subpopulations, heritability,
            global_mean,global_covariance, subpopulations[1].global_sqrt_covariance, subpopulations[1].global_inverse,
            subpopulations[1].stationary_covariance, subpopulations[1].stationary_sqrt_covariance, subpopulations[1].stationary_inverse,
            n_time, v_time, m, n_osmo, t_osmo, tau)
end

""" Copy the values A, b, omega of scr in dst. """
function copy_global_parameters!(dst::Metapopulation, scr::Metapopulation)
    dst.heritability = scr.heritability
    dst.global_mean[:] = scr.global_mean
    dst.global_covariance[:,:] = scr.global_covariance
    dst.global_inverse[:,:] = scr.global_inverse
    dst.global_sqrt_covariance[:,:] = scr.global_sqrt_covariance
    dst.stationary_covariance[:,:] = scr.stationary_covariance
    dst.stationary_inverse[:,:] = scr.stationary_inverse
    dst.stationary_sqrt_covariance[:,:] = scr.stationary_sqrt_covariance
    for k = 1:dst.n_subpopulations
      dst.subpopulations[k].heritability = dst.heritability
      dst.subpopulations[k].global_mean[:] = dst.global_mean
      dst.subpopulations[k].global_covariance[:,:] = dst.global_covariance
      dst.subpopulations[k].global_inverse[:,:] = dst.global_inverse
      dst.subpopulations[k].global_sqrt_covariance[:,:] = dst.global_sqrt_covariance
      dst.subpopulations[k].stationary_covariance[:,:] = dst.stationary_covariance
      dst.subpopulations[k].stationary_inverse[:,:] = dst.stationary_inverse
      dst.subpopulations[k].stationary_sqrt_covariance[:,:] = dst.stationary_sqrt_covariance
    end
end
