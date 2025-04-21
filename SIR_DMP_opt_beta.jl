"""Dynamic message-passing on SIR model, proposed in
Andrey Y. Lokhov et al., Phys. Rev. E 91, 012811, 2015.

At initial time, a node i is in state I with probability PI[i] = σ0[i],
while it is in state S with probability PS[i] = 1 - σ0[i].

Optimization of the SIR model by back-propagation and gradient ascent:
-> Consider the infection probabilities β as the decision variables.
-> The objective is to influence/infect as many nodes as possible.

-- Bo Li
"""

using DataFrames
using CSV
using LightGraphs
using SparseArrays
using Random
using ReverseDiff

include("GraphUtil.jl")
# using .GraphUtil


function softmax(h)
    """Softmax function.
    The input h should be a 1d vector."""
    hmax = maximum(h)
    Ph = exp.( h .- hmax )
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
    θ =  [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]
    ϕ =  [spzeros(type_of_var, size(adj_mat)...) for t in 1:T+1]

    ## Initial condition:
    for i in 1:no_of_nodes
        for n in 1:deg[i]
            j = adj_n[i, n]
            θ[1][i, j] = 1.
            ϕ[1][i, j] = σ0[i]
            PS[1][i, j] = (1 - σ0[i])
        end
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            for n in 1:deg[i]
                j = adj_n[i, n]
                temp = 1.
                for m in 1:deg[i]
                    k = adj_n[i, m]
                    if k == j
                        continue
                    end
                    θ[t][k, i] = θ[t-1][k, i] - β[k, i] * ϕ[t-1][k, i]
                    temp *= θ[t][k, i]
                end
                PS[t][i, j] = (1 - σ0[i]) * temp
                ϕ[t][i, j] = (1 - β[i, j] - μ[i] + β[i, j]*μ[i]) * ϕ[t-1][i, j] -
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

    PS_mgn = zeros(type_of_var, T+1, no_of_nodes)
    PI_mgn = zeros(type_of_var, T+1, no_of_nodes)
    PR_mgn = zeros(type_of_var, T+1, no_of_nodes)

    ## Initial condition:
    for i in 1:no_of_nodes
        PS_mgn[1, i] = 1 - σ0[i]
        PI_mgn[1, i] = σ0[i]
        PR_mgn[1, i] = 0
    end

    for t in 2:T+1
        for i in 1:no_of_nodes
            temp = 1.
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


function forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, opt)
    """Objective function governed by the forward equations."""
    PS, θ, ϕ = dynamic_MP(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, opt)
    PS_mgn, PI_mgn, PR_mgn =
        DMP_marginal(T, adj_mat, adj_n, deg, σ0, βv, μ, PS, θ, ϕ, opt)

    ## let the objective be the expected no. of nodes who have been infected:
    obj = sum( 1 .- PS_mgn[T+1, :] )
    return obj
end


function gradient_descent_over_β(edge_list, adj_mat, adj_n, deg)
    """Gradient descent over edge weigths βv[e] = β[i,j].
    Keep the best solution."""
    ## Setting the parameters:
    no_of_nodes = size(adj_mat, 1)
    no_of_edges = size(edge_list, 1)

    ## Only node 1 is infected at the initial time:
    # σ0 = zeros(Float64, no_of_nodes)
    # σ0[1] = 1.
    Random.seed!(100)
    σ0 = rand(no_of_nodes) * 0.05

    βmag = 0.1
    μmag = 0.2
    βv = ones(Float64, no_of_edges) * βmag
    μ = ones(Float64, no_of_nodes) * μmag
    βtot = sum(βv)

    T = 10

    ## objective as a function of βv:
    objf_of_βv = x -> forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0, x, μ, "beta")

    o1 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, "beta")
    βv_best = copy(βv)
    obj_best = o1

    ## Perform gradient ascent by 10 steps:
    for step in 1:10
        ∇objf = ReverseDiff.gradient(objf_of_βv, βv)
        s = 0.1                         ## step size of gradient update, a hyperparameter to tune
        βv += s*∇objf
        βv = min.(max.(βv, 0), 1)       ## to ensure the control parameter lies between 0 and 1
        βv *= βtot / sum(βv)            ## to ensure the sum of control parameters is conserved

        ## keep the best solution:
        obj = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0, βv, μ, "beta")
        if obj > obj_best
            βv_best[:] = βv[:]
            obj_best = obj
        end

        println("at step = $(step), sum_β = $(sum(βv)), |∇objf|^2=$(sum(∇objf .^ 2)).")
    end

    return o1, obj_best, βv_best, σ0
end



## Directory to dump results:
dir_result = "./results/"

## Artificial networks:
dir_network = "./artificial_networks/"
# graph_name = "bt_depth6"                ## DMP should be exact on tree networks
# graph_name = "star6"
graph_name = "ER_N100_d5_seed102"
# graph_name = "rrg_N100_d5_seed100"


node_data, edge_data, graph = GraphUtil.read_graph_from_csv(dir_network * graph_name, false)   # an un-directed graph
if !is_connected(graph)
    println("The base graph is not connected.")
end
max_deg, deg, edge_list, edge_indx, adj_n, adj_e, adj_e_indx, adj_mat, B = GraphUtil.undigraph_repr(graph, edge_data)


@time begin

o1, obj_best, βv_best, σ0 = gradient_descent_over_β(edge_list, adj_mat, adj_n, deg)
println("objf before and after opt: $(o1), $(obj_best)")


## save the solution:
open(dir_result * "betav_" * graph_name * ".csv", "w") do io
    for e in 1:ne(graph)
        write(io, "$(βv_best[e])\n")
    end
end

open(dir_result * "sigma0_" * graph_name * ".csv", "w") do io
    for i in 1:nv(graph)
        write(io, "$(σ0[i])\n")
    end
end

end  ## end of @time
