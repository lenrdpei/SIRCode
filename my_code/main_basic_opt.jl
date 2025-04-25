using Pkg
Pkg.activate("project_env")

include("GraphUtil.jl")
include("DMP_SIROpt.jl")
using .GraphUtil
using .DMP_SIROpt

using Graphs, Random, CSV, DataFrames

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

no_of_nodes = size(adj_mat, 1)
no_of_edges = size(edge_list, 1)

@time begin

## Problem parameters:
βmag = 0.2
μmag = 1.
βv = ones(Float64, no_of_edges) * βmag
μ = ones(Float64, no_of_nodes) * μmag

T = 20

λ = 1.

## All nodes to be influenced:
# PTargets = collect(1:nv(graph))
# NTargets = []

## Randomly pick some nodes to be influenced (PTargets), and some other nodes NOT to be influenced (NTargets):
Random.seed!(110)
ind = randperm( nv(graph) )
PTargets = ind[1:50]
NTargets = ind[51:100]

σtot = 5
o1, o2, σ0 = gradient_descent_over_σ0_multiseed(edge_list, adj_mat, adj_n, deg, σtot, T, βv, μ, λ, PTargets, NTargets)
set_of_seeds, σ0_soln = round_up_σ0(σ0, σtot)

o3 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0_soln, βv, μ, "sigma0", λ, PTargets, NTargets)
println("obj after round up: $(o3)\n")

## Trajectories based on σ0_soln:
PS, θ, ϕ = dynamic_MP(T, edge_list, adj_mat, adj_n, deg, σ0_soln, βv, μ, "sigma0")
PS_mgn, PI_mgn, PR_mgn =
    DMP_marginal(T, adj_mat, adj_n, deg, σ0_soln, βv, μ, PS, θ, ϕ, "sigma0")

## save the solution:
node_types = zeros(Int, no_of_nodes)
node_types[PTargets] .= 1
node_types[NTargets] .= -1
open(dir_result * "sigma0_" * graph_name * ".csv", "w") do io
    write(io, "node_id,node_type,σ0,σ0_rounded,P_S\n")
    for i in 1:no_of_nodes
        write(io, "$i,$(node_types[i]),$(σ0[i]),$(σ0_soln[i]),$(PS_mgn[T+1,i])\n")
    end
end

end  ## end of @time