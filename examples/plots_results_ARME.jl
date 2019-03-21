using LinearAlgebra
using IdentificationWithARME
using JLD
using PyCall
using LaTeXStrings
@pyimport matplotlib.pyplot as plt
include("constants.jl")
# Parameters used for the simulated data, where b is the exponential of b.
a_data = Matrix(0.5I,3,3)
b_data = [0.294, 0.947, 0.1]
h_data = 20.
omega_data = [sqrt(0.1) 0. 0.; sqrt(0.05) sqrt(0.05) 0.;sqrt(0.1/3) sqrt(0.1/3) sqrt(0.1/3)]*
    [sqrt(0.1) 0. 0.; sqrt(0.05) sqrt(0.05) 0.;sqrt(0.1/3) sqrt(0.1/3) sqrt(0.1/3)]'

# Computation of the covariance matrix of the individual parameters.
sigma_data = zeros(3,3)
for i = 1:3
    for j = 1:3
        sigma_data[i,j] = omega_data[i,j]/(1-a_data[i,i]*a_data[j,j])
    end
end

fontsize = 28
labelsize = 28


# Number of results to plot
n_rep = 20

# Dimension of the individual parameters
m = 3

# Name of the file for saving the figure
saved_file = "validation_plot_ARME.pdf"

# File path for the saved results
file_path = "./data/SAEM_nondiagcov_diagA_h20_"

# Loading of the results. FI is the Fisher Information matrix.
a, b, omega, FI = load_estim_values(file_path, 1:n_rep)

# Number of iteration in the ARME algorithm, which corresponds to the length of the result minus 1, the initial condition.
N = length(a[1]) - 1


# Plot of b in the natural (not log) domain: we take the exponential of the results.
for i = 1:n_rep
    for k = 1:N+1
        b[i][k] = exp.(b[i][k])
    end
end
sigma = copy(omega)
for k = 1:n_rep
    for k1 = 1:N+1
        for l = 1:m
            for p = 1:m
                sigma[k][k1][l,p] = omega[k][k1][l,p]/(1-a[k][k1][l,l]*a[k][k1][p,p])
            end
        end
    end
end
# Set to true in Omega is non diagonal
nondiagcov = true
################################################################################
if !nondiagcov
    fig_tot = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
else
    fig_tot = plt.subplots(nrows = 3, ncols = 5, figsize = (35,25))
end
# Set individual_traj=true to plot the trajectories for each dataset
plot_with_data_quantile(a, fig_tot, "blue", N, a_data, 1:3, m; individual_traj = false)
fig_tot[2][1][:set_title](L"$A_{1,1}$", fontsize = fontsize)
fig_tot[2][2][:set_title](L"$A_{2,2}$", fontsize = fontsize)
fig_tot[2][3][:set_title](L"$A_{3,3}$", fontsize = fontsize)
# Set individual_traj=true to plot the trajectories for each dataset
plot_with_data_quantile(b, fig_tot, "blue", N, b_data, 4:6, m; individual_traj = false)
fig_tot[2][4][:set_title](string("Pop. mean of ", L"$g_m$"), fontsize = fontsize)
fig_tot[2][5][:set_title](string("Pop. mean of ", L"$k_p$"), fontsize = fontsize)
fig_tot[2][6][:set_title](string("Pop. mean of ", L"$g_p$"), fontsize = fontsize)
if !nondiagcov
    # Set individual_traj=true to plot the trajectories for each dataset
    plot_with_data_quantile(sigma, fig_tot, "blue", N, sigma_data, 7:9, m; individual_traj=false)
    legende = [L"$\Sigma_{1,1}$", L"$\Sigma_{2,2}$", L"$\Sigma_{3,3}$"]
    for i = 7:9
        fig_tot[2][i][:set_title](legende[i-6], fontsize = fontsize)
    end
    for j = 1:9
        if j<4
            fig_tot[2][j][:set_ylim](0.,0.8)
        end
        fig_tot[2][j][:yaxis][:set_tick_params](pad = 5, length = 5, labelsize=labelsize)
        if j in [3,6,9]
            fig_tot[2][j][:xaxis][:set_tick_params](pad = 5, length = 5, labelsize=labelsize)
            fig_tot[2][j][:set_xlabel]("# iterations", fontsize = fontsize)
        else
            fig_tot[2][j][:xaxis][:set_visible](false)
        end
    end
else
    # Set individual_traj=true to plot the trajectories for each dataset
    plot_with_data_quantile(sigma, fig_tot, "blue", N, sigma_data, 7:15, m;  nondiagcov=true, individual_traj=false)
    legende = [L"$\Sigma_{1,1}$", L"$\Sigma_{2,1}$", L"$\Sigma_{3,1}$",
        L"$\Sigma_{1,2}$", L"$\Sigma_{2,2}$", L"$\Sigma_{3,2}$",
        L"$\Sigma_{1,3}$", L"$\Sigma_{2,3}$", L"$\Sigma_{3,3}$"]
    for i = 7:15
        fig_tot[2][i][:set_title](legende[i-6], fontsize = fontsize)
    end
    for j = 1:15
        if j<4
            fig_tot[2][j][:set_ylim](0.,0.8)
        end
        fig_tot[2][j][:yaxis][:set_tick_params](pad = 5, length = 5, labelsize=labelsize)
        if j in [3,6,9,12,15]
            fig_tot[2][j][:xaxis][:set_tick_params](pad = 5, length = 5, labelsize=labelsize)
            fig_tot[2][j][:set_xlabel]("# iterations", fontsize = fontsize)
        else
            fig_tot[2][j][:xaxis][:set_visible](false)
        end
    end
end
fig_tot[1][:savefig](saved_file)
plt.show()
