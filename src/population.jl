
mutable struct Population{HeritabilityT}
    size::Int
    individual_parameters::Array{Float64,2}
    ancestor_index::Array{Int,1}
    offspring::Array{Array{Int,1},1}
    n_parameters::Int
    exp_individual_parameters::Array{Float64,2}
    n_time::Int
    v_time::Array{Float64,1}
    birth_death_time::Array{Float64,2} # for each individual (column) birth time and death time
    individual_index_time::Array{Bool,2} # for each individual (column), true = the individual is alive at this time
    first_sampling::Array{Int,1} # index of the first sampling for each individual
    protein_density::Array{Float64,2}
    protein_density_beginning::Array{Float64,1} # starting value of protein density for each individual
    mRNA_density_beginning::Array{Float64,1} # starting value of mRNA density for each individual
    Hog1_beginning::Array{Float64,1} # starting value of Hog1 activity for each individual
    heritability::HeritabilityT
    global_mean::Array{Float64,1}
    global_covariance::Array{Float64,2}
    global_sqrt_covariance::Array{Float64,2}
    global_inverse::Array{Float64,2}
    stationary_covariance::Array{Float64,2}
    stationary_sqrt_covariance::Array{Float64,2}
    stationary_inverse::Array{Float64,2}
    n_osmo_global::Int #  total number of osmotic shocks
    t_osmo_global::Array{Float64,2} # ith line : the ith osmotic shock, first column: beginning, second : end
    t_osmo_beginning::Array{Array{Float64,1},1} # first index :  individual, second index : sequence of the beginning times of osmotic shocks during the life of the individual
    t_osmo_end::Array{Array{Float64,1},1} #  first index :  individual, second index : sequence of the ending times of osmotic shocks during the life of the individual
    area_osmo::Array{Int,2} # area_osmo[i] = number of area in which we are at time v_time[i] (area 1 : before first shock, area 2: during first shock, area 3 : between shock 1 and 2 ...), depend on the time of life of each individual
    n_gene::Int # max. number of generation
    tau::Float64 # delay for fluorescence
end


function stationnary_covariance_fun(heritability::Array{Float64,2}, global_covariance::Array{Float64,2})
    stationnary_covariance = zeros(Float64,3,3)
    for i = 1:3
        for j= 1:3
            stationnary_covariance[i,j] = global_covariance[i,j]/(1-heritability[i,i]*heritability[j,j])
        end
    end
    return stationnary_covariance
end

""" Create a population from size, ancestor_index, population parameters and indications on osmotic shocks. If test_maximization = true, do not compute the offsring array ;
 index_time : specify the indices in v_time where an individual is observed, default: each time between birth and death. """
Population(n::Int64, ancestor_index::Array{Int64,1}, heritability::HeritabilityT,
        global_mean::Array{Float64, 1}, global_covariance::Array{Float64,2},
        n_time::Int64, v_time::Union{AbstractRange,Array{Float64,1}},
        birth_death_time::Array{Float64, 2}, m::Int64, n_osmo::Int64,
        t_osmo::Array{Float64, 2}, tau::Float64; test_maximization=false, index_time=0) where HeritabilityT<:Union{Float64, Array{Float64,2}} =
begin
   individual_parameters = zeros(Float64,m,n)  # en colonne (gm,kp,gp)
   exp_individual_parameters = zeros(Float64,m,n)
   protein_density =  NaN*zeros(Float64,n_time,n)
   protein_density_beginning = zeros(Float64,n)
   mRNA_density_beginning =  zeros(Float64,n)
   Hog1_beginning = zeros(Float64,n)
   global_sqrt_covariance = cholesky(global_covariance).U'
   global_inverse = global_covariance^(-1)
   stationary_covariance  = stationnary_covariance_fun(heritability, global_covariance)
   stationary_sqrt_covariance = cholesky(stationary_covariance).U'
   stationary_inverse = stationary_covariance^(-1)
   c_osmo = zeros(n_osmo+1,n) # sum for i =1 to k-1 -> 0 for the first value
   individual_osmotic_shock = zeros(Bool,n_osmo,n)
   individual_index_time = zeros(Bool,n_time,n)
   first_sampling = zeros(Int,n)
   for u = 1:n
       if index_time == 0
           individual_index_time[:,u] = (birth_death_time[1,u] .<= v_time .<= birth_death_time[2,u])
       else
           individual_index_time[:,u] = index_time[:,u]
       end
       first_sampling[u] = findfirst(individual_index_time[:,u])
   end
   t_osmo_beginning = Array{Array{Float64,1},1}()
   t_osmo_end = Array{Array{Float64,1},1}()
   for u = 1:n
       x = (birth_death_time[1,u] .<= t_osmo[:,2]) .& (birth_death_time[2,u] .>= t_osmo[:,1])
       push!(t_osmo_end, t_osmo[x,2])
       push!(t_osmo_beginning, t_osmo[x,1])
   end
   area_osmo = -1*zeros(n_time,n)
   index_current::BitArray{1} = falses(n_time)
   for u = 1:n
       index_current = individual_index_time[:,u]
       if length(t_osmo_beginning[u]) == 0
           area_osmo[index_current,u] = ones(sum(index_current))
       else
           area_osmo[index_current,u] = sum((v_time[index_current] .>= t) for t in t_osmo_beginning[u]) .+
           sum((v_time[index_current] .> t) for t in t_osmo_end[u]) .+ 1
       end
   end
   offspring::Array{Array{Int64,1},1} = []
   if !test_maximization
       for u = 1:n
           queue = [u]
           offspring_current = [u]
           while length(queue)>0
               offspring_current = vcat(offspring_current, findall(ancestor_index.== queue[1]))
               queue = vcat(queue, findall(ancestor_index.== queue[1]))[2:end]
           end
           push!(offspring, offspring_current)
       end
   end
   Population{typeof(heritability)}(n, individual_parameters,ancestor_index, offspring, m, exp_individual_parameters,n_time,
   v_time, birth_death_time, individual_index_time, first_sampling, protein_density, protein_density_beginning,
   mRNA_density_beginning, Hog1_beginning, heritability, global_mean, global_covariance, global_sqrt_covariance,
   global_inverse, stationary_covariance, stationary_sqrt_covariance, stationary_inverse, n_osmo, t_osmo,
   t_osmo_beginning, t_osmo_end, area_osmo, floor(log2(n))+1, tau)
end

""" Exchange the individual parameters of pop1 and pop2. """
function swap_parameters!(pop1::Population, pop2::Population)
    tmp_individual_parameters = pop1.individual_parameters
    tmp_exp_individual_parameters = pop1.exp_individual_parameters
    tmp_protein_density = pop1.protein_density
    tmp_protein_density_beginning = pop1.protein_density_beginning
    tmp_mRNA_density_beginning = pop1.mRNA_density_beginning

    pop1.individual_parameters = pop2.individual_parameters
    pop1.exp_individual_parameters = pop2.exp_individual_parameters
    pop1.protein_density = pop2.protein_density
    pop1.protein_density_beginning = pop2.protein_density_beginning
    pop1.mRNA_density_beginning = pop2.mRNA_density_beginning

    pop2.individual_parameters = tmp_individual_parameters
    pop2.exp_individual_parameters = tmp_exp_individual_parameters
    pop2.protein_density = tmp_protein_density
    pop2.protein_density_beginning = tmp_protein_density_beginning
    pop2.mRNA_density_beginning = tmp_mRNA_density_beginning
end
"""
Copy the values of parameters of src into dst for u and it descendants. Default u=1.
"""
function copy_some_parameters!(dst::Population, src::Population ; u=1)
    dst.protein_density[:,u] = src.protein_density[:,u]
    dst.individual_parameters[:,u] = src.individual_parameters[:,u]
    dst.exp_individual_parameters[:,u] = src.exp_individual_parameters[:,u]
    for v in src.offspring[u][2:end]
        dst.protein_density[:,v] = src.protein_density[:,v]
        dst.mRNA_density_beginning[v] = src.mRNA_density_beginning[v]
        dst.protein_density_beginning[v] = src.protein_density_beginning[v]
        dst.individual_parameters[:,v] = src.individual_parameters[:,v]
        dst.exp_individual_parameters[:,v] = src.exp_individual_parameters[:,v]
    end
end
