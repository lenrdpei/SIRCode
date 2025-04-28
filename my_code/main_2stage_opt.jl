include("GraphUtil.jl")
include("BetaGen.jl")
include("DMP_SIROpt.jl")
include("MLP4Beta.jl")
using .GraphUtil
using .BetaGen
using .DMP_SIROpt
using .MLP4Beta

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
# βmag = 0.2
# βv = ones(Float64, no_of_edges) * βmag

# -----------------------------------------------------------------
# Generate design matrx Z and ground truth βv
# -----------------------------------------------------------------
feature_p = 20
instance_num = 200
problem_deg = 3
noise_sigma = 1.0
offset_c = -1.5

B = generate_B(feature_p; use_continuous_B=false)

Z = zeros(Float64, no_of_nodes, feature_p, instance_num)
β_grd = zeros(Float64, no_of_nodes, instance_num)
βv_grd = zeros(Float64, no_of_edges, instance_num)
for _i in 1:instance_num
    # Z, β_node = generate_beta(B, no_of_nodes; p=feature_p, deg=2, σ=0.1)
    Z[:, :, _i], β_grd[:, _i] = generate_beta(B, no_of_nodes; p=feature_p, deg=problem_deg, σ=noise_sigma, offset_c=offset_c)
    βv_grd[:, _i] = β_to_βv(β_grd[:, _i], edge_list)  # convert node vector to edge vector
    # for e in 1:no_of_edges
    #     i, j = edge_list[e, :]
    #     βv_grd[e, _i] = (β_grd[i, _i] + β_grd[j, _i]) / 2.0
    # end
end

# -----------------------------------------------------------------
# Predict βv with Z using MLP
# -----------------------------------------------------------------
# train-test split:
train_ratio = 0.5
train_size = Int(train_ratio * instance_num)
test_size = instance_num - train_size

Z_train = Z[:, :, 1:train_size]
Z_test = Z[:, :, (train_size+1):instance_num]
β_train = β_grd[:, 1:train_size]
β_test = β_grd[:, (train_size+1):instance_num]
βv_train = βv_grd[:, 1:train_size]
βv_test = βv_grd[:, (train_size+1):instance_num]

# prediction with MLP:
model = train_mlp(Float32.(Z_train), Float32.(β_train); epochs=Int(10000/(instance_num*(1-train_ratio))), lr=0.001)
β_pred = [model(Z_test[:, :, i]') for i in 1:test_size]
β_pred = vcat(β_pred...)'  # (|V|, test_size)
βv_pred = zeros(Float64, no_of_edges, test_size)
for i in 1:test_size
    βv_pred[:, i] = β_to_βv(β_pred[:, i], edge_list)  # convert node vector to edge vector
end
# -----------------------------------------------------------------
# Specify other problem parameters
# -----------------------------------------------------------------
μmag = 1.
μ = ones(Float64, no_of_nodes) * μmag

T = 100
λ = 1.
σtot = 5
# All nodes to be influenced:
PTargets = collect(1:nv(graph))
NTargets = []

## Randomly pick some nodes to be influenced (PTargets), and some other nodes NOT to be influenced (NTargets):
# Random.seed!(110)
# ind = randperm( nv(graph) )
# PTargets = ind[1:50]
# NTargets = ind[51:100]

# -----------------------------------------------------------------
# Maximize the objective function w.r.t. σ0
o1_vec = zeros(Float64, test_size)
o2_vec = zeros(Float64, test_size)
o3_vec = zeros(Float64, test_size)
o4_vec = zeros(Float64, test_size)
o5_vec = zeros(Float64, test_size)
for _i in 1:test_size
    βv = βv_pred[:, _i]
    βv_truth = βv_test[:, _i]
    # o1: objective with randomly initialized σ0; o2: objective after optimization
    o1, o2, σ0 = gradient_descent_over_σ0_multiseed(edge_list, adj_mat, adj_n, deg, σtot, T, βv, μ, λ, PTargets, NTargets; verbose=false)
    set_of_seeds, σ0_soln = round_up_σ0(σ0, σtot)
    println("o1 with random σ0 and β hat: $(o1)")
    println("o2 with optimized σ0 and β hat: $(o2)")

    # o3: objective with rounded σ0(β hat) and β hat
    o3 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0_soln, βv, μ, "sigma0", λ, PTargets, NTargets)
    println("o3 with rounded σ0(β hat) and β hat: $(o3)")

    # o4: objective with rounded σ0(β hat) and β truth
    o4 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0_soln, βv_truth, μ, "sigma0", λ, PTargets, NTargets)
    println("o4 with rounded σ0(β hat) and β truth: $(o4)")

    _, _, σ0_truth = gradient_descent_over_σ0_multiseed(edge_list, adj_mat, adj_n, deg, σtot, T, βv_truth, μ, λ, PTargets, NTargets; verbose=false)
    _, σ0_soln_truth = round_up_σ0(σ0_truth, σtot)
    # o5: objective with rounded σ0(β truth) and β truth
    o5 = forward_obj_func(T, edge_list, adj_mat, adj_n, deg, σ0_soln_truth, βv_truth, μ, "sigma0", λ, PTargets, NTargets)
    println("o5 with rounded σ0(β truth) and β truth: $(o5)")

    regret = o5 - o4
    println("regret o5 - o4: $(regret)")

    ## Trajectories based on σ0_soln:
    PS, θ, ϕ = dynamic_MP(T, edge_list, adj_mat, adj_n, deg, σ0_soln, βv, μ, "sigma0")
    PS_mgn, PI_mgn, PR_mgn =
        DMP_marginal(T, adj_mat, adj_n, deg, σ0_soln, βv, μ, PS, θ, ϕ, "sigma0")

    # save objective values:
    o1_vec[_i] = o1
    o2_vec[_i] = o2
    o3_vec[_i] = o3
    o4_vec[_i] = o4
    o5_vec[_i] = o5
    # ## save the solution:
    # node_types = zeros(Int, no_of_nodes)
    # open(dir_result * "sigma0_" * graph_name * ".csv", "w") do io
    #     write(io, "node_id,node_type,σ0,σ0_rounded,P_S\n")
    #     for i in 1:no_of_nodes
    #         write(io, "$i,$(node_types[i]),$(σ0[i]),$(σ0_soln[i]),$(PS_mgn[T+1,i])\n")
    #     end
    # end
    # open(dir_result * graph_name * "/test_$(_i)" * ".csv", "w") do io
    #     write(io, "node_id,node_type,σ0,σ0_rounded,P_S\n")
    #     for i in 1:no_of_nodes
    #         write(io, "$i,$(node_types[i]),$(σ0[i]),$(σ0_soln[i]),$(PS_mgn[T+1,i])\n")
    #     end
    # end
end
open(dir_result * graph_name * "_obj.csv", "w") do io
    write(io, "o1,o2,o3,o4,o5\n")
    for i in 1:test_size
        write(io, "$(o1_vec[i]),$(o2_vec[i]),$(o3_vec[i]),$(o4_vec[i]),$(o5_vec[i])\n")
    end
end

end  ## end of @time