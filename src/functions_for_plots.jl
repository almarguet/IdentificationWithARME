"""
Return the estimated values of `A`, `b` and `Omega` from `file_index` results saved and stored in `file_path`.
"""
function load_estim_values(file_path, files_index)
    a = Array{Array{Array{Float64,2},1},1}()
    b = Array{Array{Array{Float64,1},1},1}()
    omega = Array{Array{Array{Float64,2},1},1}()
    FI = Array{Array{Array{Float64,2},1},1}()
    for k in files_index
        res = load(string(file_path,k,".jld"))
        push!(a, res["heritability"])
        push!(b, res["mean"])
        push!(omega, res["omega"])
        push!(FI, res["FI"])
    end
    return a, b, omega, FI
end

""" Compute the mean, standard deviation and if compute_quantile = true, the quantiles, of the estimated values estim_values
 obtained from repeated experiments. Case of a estimated value of type Array{Float64,2}. """
function statistics_for_plots(estim_values::Array{Array{Array{Float64,2},1},1}, N, m; compute_quantile=false)
    estim_values_stat = zeros(length(estim_values),N+1, m, m)
    for i = 1:N+1
      for k = 1:length(estim_values)
        estim_values_stat[k,i,:,:] = estim_values[k][i][:,:]
      end
    end
    if compute_quantile
        res = zeros(m, N+1, m, m)
        for i = 1:N+1
            for j = 1:m
                for l = 1:m
                    res[:, i,j,l] = quantile(estim_values_stat[:,i,j,l], [0.25, 0.5, 0.75])
                end
            end
        end
        return res
    else
        return mean(estim_values_stat ; dims = 1), std(estim_values_stat ; dims = 1)
    end
end
""" Compute the mean, standard deviation and if compute_quantile = true, the quantiles, of the estimated values estim_values
 obtained from repeated experiments. Case of a estimated value of type Array{Float64,1}. """
function statistics_for_plots(estim_values::Array{Array{Array{Float64,1},1},1}, N, m; compute_quantile=false)
    estim_values_stat = zeros(length(estim_values),N+1, m)
    for i = 1:N+1
      for k = 1:length(estim_values)
        estim_values_stat[k,i,:] = estim_values[k][i][:]
      end
    end
    if compute_quantile
        res = zeros(m, N+1, m)
        for i = 1:N+1
            for j = 1:m
                res[:, i,j] = quantile(estim_values_stat[:,i,j], [0.25, 0.5, 0.75])
            end
        end
        return res
    else
        return mean(estim_values_stat ; dims = 1), std(estim_values_stat ; dims = 1)
    end
end

"""
Plots for `A` or `Omega`.

# Arguments

-`estim_values`: Array of results of the algorithm for different datasets.

-`fig`: figure for the plot.

-`col`: color of the curves and intervals.

-`N`: number of ARME iterations.

-`data_values`: target of the estimation.

-`index_plot`: indices of fig for the corresponding plot.

-`m`: dimension of the individual parameters.

# Keyword Arguments

-`nondiagcov`: default_value=false. Set true if Omega is non diagonal.

-`individual_traj`: default_value=false. Set true to plot the trajectories of the algorithm for each dataset.

"""
function plot_with_data_quantile(estim_values::Array{Array{Array{Float64,2},1},1}, fig, col, N, data_values, index_plot, m;  nondiagcov=false, individual_traj=false)
    quantile_values = statistics_for_plots(estim_values, N, m; compute_quantile=true)
    upper_whis_bound = quantile_values[3,:,:,:] + 1.5*(quantile_values[3,:,:,:] - quantile_values[1,:,:,:])
    lower_whis_bound = quantile_values[1,:,:,:] - 1.5*(quantile_values[3,:,:,:] - quantile_values[1,:,:,:])
    if nondiagcov
        for j in index_plot
            if individual_traj
                for i = 1:length(estim_values)
                    fig[2][j][:plot](range(0,stop = N,length = N+1),[estim_values[i][k][j-6] for k = 1:N+1], color = col , alpha = 0.1)
                end
            end
            upper_whis = - Inf*ones(N+1)
            lower_whis = Inf*ones(N+1)
            for k = 1:N+1
                for l = 1:length(estim_values)
                    if estim_values[l][k][j-6]< upper_whis_bound[k,:,:][j-6]
                        upper_whis[k] = max(upper_whis[k], estim_values[l][k][j-6])
                    end
                    if estim_values[l][k][j-6]> lower_whis_bound[k,:,:][j-6]
                        lower_whis[k] = min(lower_whis[k], estim_values[l][k][j-6])
                    end
                end
            end
            fig[2][j][:plot](range(0,stop = N,length = N+1),quantile_values[2,:, mod(j-4,3)+1, div(j-4,3)], color = col)
            fig[2][j][:fill_between](range(0,stop = N,length = N+1), quantile_values[1,:, mod(j-4,3)+1, div(j-4,3)],
                quantile_values[3,:, mod(j-4,3)+1, div(j-4,3)], facecolor = col, alpha  = 0.5)
            fig[2][j][:hlines](data_values[j-6], 0, N)
            fig[2][j][:plot](range(0,stop = N,length = N+1), lower_whis, linestyle = "dotted", color = "black")
            fig[2][j][:plot](range(0,stop = N,length = N+1), upper_whis, linestyle = "dotted", color = "black")
        end
    else
        for j in index_plot
            if individual_traj
                for i = 1:length(estim_values)
                    fig[2][j][:plot](range(0,stop = N,length = N+1),[estim_values[i][k][j,j] for k = 1:N+1], color = col , alpha = 0.1)
                end
            end
            upper_whis = - Inf*ones(N+1)
            lower_whis = Inf*ones(N+1)
            for k = 1:N+1
                for l = 1:length(estim_values)
                    if estim_values[l][k][mod(j+2,3)+1,mod(j+2,3)+1]< upper_whis_bound[k,mod(j+2,3)+1,mod(j+2,3)+1]
                        upper_whis[k] = max(upper_whis[k], estim_values[l][k][mod(j+2,3)+1,mod(j+2,3)+1])
                    end
                    if estim_values[l][k][mod(j+2,3)+1,mod(j+2,3)+1]> lower_whis_bound[k,mod(j+2,3)+1,mod(j+2,3)+1]
                        lower_whis[k] = min(lower_whis[k], estim_values[l][k][mod(j+2,3)+1,mod(j+2,3)+1])
                    end
                end
            end
            fig[2][j][:plot](range(0,stop = N,length = N+1),quantile_values[2,:,mod(j+2,3)+1, mod(j+2,3)+1], color = col)
            fig[2][j][:fill_between](range(0,stop = N,length = N+1), quantile_values[1,:, mod(j+2,3)+1, mod(j+2,3)+1],
                quantile_values[3,:, mod(j+2,3)+1, mod(j+2,3)+1], facecolor = col, alpha  = 0.5)
            fig[2][j][:hlines](data_values[mod(j+2,3)+1,mod(j+2,3)+1], 0, N)
            fig[2][j][:plot](range(0,stop = N,length = N+1), lower_whis, linestyle = "dotted", color = "black")
            fig[2][j][:plot](range(0,stop = N,length = N+1), upper_whis, linestyle = "dotted", color = "black")
        end
    end
end
"""
Plots for `b`.

# Arguments

-`estim_values`: Array of results of the algorithm for different datasets.

-`fig`: figure for the plot.

-`col`: color of the curves and intervals.

-`N`: number of ARME iterations.

-`data_values`: target of the estimation.

-`index_plot`: indices of fig for the corresponding plot.

-`m`: dimension of the individual parameters.

# Keyword Arguments

-`nondiagcov`: default_value=false. Set true if Omega is non diagonal.

-`individual_traj`: default_value=false. Set true to plot the trajectories of the algorithm for each dataset.

"""
function plot_with_data_quantile(estim_values::Array{Array{Array{Float64,1},1},1},fig, col, N, data_values, index_plot, m; individual_traj=false)
    quantile_values = statistics_for_plots(estim_values, N, m; compute_quantile=true)
    upper_whis_bound = quantile_values[3,:,:] + 1.5*(quantile_values[3,:,:] - quantile_values[1,:,:])
    lower_whis_bound = quantile_values[1,:,:] - 1.5*(quantile_values[3,:,:] - quantile_values[1,:,:])
    for j in index_plot
        if individual_traj
            for i = 1:length(estim_values)
                fig[2][j][:plot](range(0,stop = N,length = N+1),[estim_values[i][k][j] for k = 1:N+1], color = col , alpha = 0.1)
            end
        end
        upper_whis = - Inf*ones(N+1)
        lower_whis = Inf*ones(N+1)
        for k = 1:N+1
            for l = 1:length(estim_values)
                if estim_values[l][k][mod(j+2,3)+1]< upper_whis_bound[k,mod(j+2,3)+1]
                    upper_whis[k] = max(upper_whis[k], estim_values[l][k][mod(j+2,3)+1])
                end
                if estim_values[l][k][mod(j+2,3)+1]> lower_whis_bound[k,mod(j+2,3)+1]
                    lower_whis[k] = min(lower_whis[k], estim_values[l][k][mod(j+2,3)+1])
                end
            end
        end
        fig[2][j][:plot](range(0,stop = N,length = N+1),quantile_values[2,:, mod(j+2,3)+1], color = col)
        fig[2][j][:fill_between](range(0,stop = N,length = N+1), quantile_values[1,:, mod(j+2,3)+1],
            quantile_values[3,:, mod(j+2,3)+1], facecolor = col, alpha  = 0.5)
        fig[2][j][:hlines](data_values[mod(j+2,3)+1], 0, N)
        fig[2][j][:plot](range(0,stop = N,length = N+1), lower_whis, linestyle = "dotted", color = "black")
        fig[2][j][:plot](range(0,stop = N,length = N+1), upper_whis, linestyle = "dotted", color = "black")
    end
end
