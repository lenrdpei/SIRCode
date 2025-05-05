module DMP_SIROpt
"""Dynamic message-passing on SIR model, proposed in
Andrey Y. Lokhov et al., Phys. Rev. E 91, 012811, 2015.

At initial time, a node i is in state I with probability PI[i] = σ0[i],
while it is in state S with probability PS[i] = 1 - σ0[i].

Optimization of the SIR model by back-propagation and gradient ascent:
-> Consider σ0 as the decision variables.
-> The objective is to select the initial seed to influence the positive targeted nodes,
   while trying to avoid influencing the negative targeted nodes.

-- Bo Li
"""

using DataFrames
using CSV
using Graphs
using SparseArrays
using Random
using ReverseDiff

export dynamic_MP, DMP_marginal, forward_obj_func, gradient_descent_over_σ0_multiseed,
    round_up_σ0

function softmax(h)
    """Softmax function.
    The input h should be a 1d vector."""
    hmax = maximum(h)
    Ph = exp.(h .- hmax)
    Ph = Ph ./ sum(Ph)
    return Ph
end


function inverse_softmax(Ph)
    """Inverse softmax function.
    The input Ph should be a 1d vector, satisfying sum(Ph)=1, 0 < Ph[i] < 1.
    Note that Ph[i] cannot be equal to 0 or 1."""
    h = log.(Ph)   ## generally h = log.(Ph) + c, where c is an arbitrary constant
    return h
end


function dynamic_MP(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, opt)
    """Dynamic message passing.

    βv is a vector containing the infection probabilities with βv[e] = β[i, j],
    where the e-th edge is (i, j).

    opt is a string indicating which variables are the control variables; this is important
    as the type of the control variables need to propagate to the dynamical variables
    for the effectiveness of autodiff."""

    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "sigma0"
        type_of_var = eltype(σ0)
    end

    no_of_nodes = size(adj_mat, 1)
    no_of_edges = size(edge_list, 1)

    β = spzeros(eltype(βv), size(adj_mat)...)   ## N X N sparse matrix

    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        β[i, j] = βv[e]
        β[j, i] = βv[e]
    end

    PS = [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]
    θ = [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]
    ϕ = [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]

    ## Initial condition:
    for i in 1:no_of_nodes
        for n in 1:deg[i]
            j = adj_n[i, n]
            θ[1][i, j] = 1.0
            ϕ[1][i, j] = σ0[i]
            PS[1][i, j] = (1 - σ0[i])
        end
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            for n in 1:deg[i]
                j = adj_n[i, n]
                temp = 1.0
                for m in 1:deg[i]
                    k = adj_n[i, m]
                    if k == j
                        continue
                    end
                    θ[t][k, i] = θ[t-1][k, i] - β[k, i] * ϕ[t-1][k, i]
                    temp *= θ[t][k, i]
                end
                PS[t][i, j] = (1 - σ0[i]) * temp
                ϕ[t][i, j] = (1 - β[i, j] - μ[i] + β[i, j] * μ[i]) * ϕ[t-1][i, j] -
                             (PS[t][i, j] - PS[t-1][i, j])
                θ[t][i, j] = θ[t-1][i, j] - β[i, j] * ϕ[t-1][i, j]
            end
        end
    end

    return PS, θ, ϕ
end


function DMP_marginal(T, adj_mat, adj_n, deg, σ0, βv, μ, PS, θ, ϕ, opt)
    """Marginal probability of DMP.

    'opt' is a string indicating which variables are the control variables; this is important
    as the type of the control variables need to propagate to the dynamical variables
    for the effectiveness of autodiff."""

    if opt == "beta"
        type_of_var = eltype(βv)
    elseif opt == "mu"
        type_of_var = eltype(μ)
    elseif opt == "sigma0"
        type_of_var = eltype(σ0)
    end

    no_of_nodes = size(adj_mat, 1)

    PS_mgn = zeros(type_of_var, T + 1, no_of_nodes)
    PI_mgn = zeros(type_of_var, T + 1, no_of_nodes)
    PR_mgn = zeros(type_of_var, T + 1, no_of_nodes)

    ## Initial condition:
    for i in 1:no_of_nodes
        PS_mgn[1, i] = 1 - σ0[i]
        PI_mgn[1, i] = σ0[i]
        PR_mgn[1, i] = 0
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            temp = 1.0
            for m in 1:deg[i]
                k = adj_n[i, m]
                temp *= θ[t][k, i]
            end
            PS_mgn[t, i] = (1 - σ0[i]) * temp
            PR_mgn[t, i] = PR_mgn[t-1, i] + μ[i] * PI_mgn[t-1, i]
            PI_mgn[t, i] = 1 - PS_mgn[t, i] - PR_mgn[t, i]
        end
    end

    return PS_mgn, PI_mgn, PR_mgn
end


function forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, opt,
    λ, PTargets, NTargets)
    """Objective function governed by the forward equations.
    PTargets: the set of nodes to be influenced.
    NTargets: the set of nodes Not to be influenced."""
    PS, θ, ϕ = dynamic_MP(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, opt)
    PS_mgn, PI_mgn, PR_mgn =
        DMP_marginal(T, adj_mat, adj_n, deg, σ0, βv, μ, PS, θ, ϕ, opt)

    obj = sum(1 .- PS_mgn[T+1, :])
    return obj
end


function gradient_descent_over_σ0_multiseed(edge_list, adj_mat, adj_n, deg, σtot, T, βv, μ,
    λ, PTargets, NTargets; verbose=true, max_iter=60)
    """Gradient descent over initial seed σ0[i], assuming sum(σ0) = σtot.

    In this experiment, a reparameterization method is use to enforce the constraint
        sum(σ0) = σtot, 0 < σ0[i], as use in [Saad and Lokhov PNAS2017],
    i.e., we let σ0[i] = σtot* exp(h0[i]) / (sum( exp.(h0) )), i.e., σ0 = σtot * softmax(h0).

    A barrier function ϵ*log(1 - σ0[i]) is used to deal with the upper bound constraint of σ0[i] < 1.
    """
    ## Setting the parameters:
    no_of_nodes = size(adj_mat, 1)
    no_of_edges = size(edge_list, 1)

    # σ0 = ones(no_of_nodes) / no_of_nodes * σtot
    Random.seed!(100)
    σ0 = rand(no_of_nodes)
    σ0 *= σtot / sum(σ0)

    h0 = inverse_softmax(σ0 / σtot)

    ## Augmented objective function: L = objf + ϵ * ∑_i log(1 - σ0[i]), as a function of h0
    ϵ = 0.02
    L_of_h0 = x -> forward_obj_func(T, edge_list, adj_mat, adj_n, deg, softmax(x) * σtot, βv, μ, "sigma0", λ, PTargets, NTargets) +
                   ϵ * sum(log.(1 .- clamp.(softmax(x) * σtot, eps(), 1 - eps())))

    o1 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, softmax(h0) * σtot, βv, μ, "sigma0", λ, PTargets, NTargets)
    L = L_of_h0(h0)

    ## Perform gradient ascent:
    for step in 1:max_iter
        ∇L = ReverseDiff.gradient(L_of_h0, h0)

        ## Backtracking line search:
        α, γ = 0.3, 0.6     ## parameters for line search
        s = 20.0             ## initial guess of step size
        L_temp = 0.0
        for inner_step in 1:20
            h0_temp = h0 + s * ∇L
            σ0_temp = softmax(h0_temp) * σtot
            if maximum(σ0_temp) >= 1    ## To ensure σ0[i] < 1
                s *= γ
                continue
            end

            L_temp = L_of_h0(h0 + s * ∇L)
            if L_temp > L + α * s * sum(∇L .^ 2)
                L = L_temp
                break
            else
                s *= γ
            end
        end

        h0 += s * ∇L
        σ0 = softmax(h0) * σtot
        ∇L_norm = sum(∇L .^ 2)
        if verbose
            println("At step = $(step), s = $(s), min_σ0 = $(minimum(σ0)), max_σ0 = $(maximum(σ0)), |∇L|^2 = $(∇L_norm), L = $(L_temp).")
        end

        if ∇L_norm < 1e-6
            break
        end

    end

    o2 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, softmax(h0) * σtot, βv, μ, "sigma0", λ, PTargets, NTargets)

    if verbose
        println("\n")
        println("minimum and maximum of σ0: $(minimum(σ0)), $(maximum(σ0))\n")
        println("sum σ0 before and after optimization: $(σtot), $(sum(σ0))\n")
        # println("obj before and after optimization: $(o1), $(o2)\n")
    end

    return o1, o2, σ0
end


function round_up_σ0(σ0, σtot::Int)
    """Pick the nodes with highest value of σ0 as initial seeds."""
    ind = sortperm(σ0, rev=true)
    set_of_seeds = ind[1:σtot]
    σ0_soln = zeros(length(σ0))
    σ0_soln[set_of_seeds] .= 1         ## set the σ0 value of the seeding nodes to be 1
    return set_of_seeds, σ0_soln
end


end # module DMP_SIROpt