module BetaGen

using DataFrames, Random, LinearAlgebra, Distributions

export generate_B, generate_beta, β_to_βv

"""
    generate_B(p::Int; use_continuous_B::Bool = false) -> Vector{Float64}

Generates a random vector B of length p, where each element is either 0 or 1.
The vector can be optionally scaled by a random vector of length p.

# Arguments
- `p::Int`: The length of the vector B.
- `use_continuous_B::Bool`: If true, the elements of B are scaled by a random vector of length p.
    
    Default is false.

# Returns
- `Vector{Float64}`: A vector of length p, where each element is either 0 or 1.
    If `use_continuous_B` is true, the elements are scaled by a random vector of length p.
"""
function generate_B(
    p::Int;           # feature dimension
    use_continuous_B::Bool=false
)
    B = rand(Bernoulli(0.5), p)                 # Discrete B
    if use_continuous_B
        β_vec = randn(p)               # Heterogeneous scaling
        B = B .* β_vec                          # Element-wise scaling (broadcasted)
    end

    return B
end

"""
    generate_beta(pattern_B::Vector, num_nodes::Int; p::Int = 10, deg::Int = 2, σ::Float64 = 0.1) -> Tuple{Matrix{Float64}, Vector{Float64}}

Generates a random vector β of length |V|, where |V| is the number of nodes in the graph.
The vector is generated based on a design matrix Z and a random vector B.
The generation process includes a non-linear transformation and additive Gaussian noise.


# Arguments
- `pattern_B::Vector`: The underlying pattern (matrix B) used for generating β.
- `num_nodes::Int`: The number of nodes in the graph (|V|).
- `p::Int`: The feature dimension (default is 10).
- `deg::Int`: The degree of non-linearity (default is 2).
- `σ::Float64`: The standard deviation of the additive Gaussian noise (default is 0.1).


# Returns
- `Z::Matrix{Float64}`: The design matrix of shape (|V|, p).
- `β::Vector{Float64}`: The generated vector of shape (|V|,).
    The elements of β are in the range [0, 1].
"""
function generate_beta(
    pattern_B::Vector, # underlying pattern (matrix B)
    num_nodes::Int;    # |V|
    p::Int=10,       # feature dimension
    deg::Int=2,      # non-linearity degree
    σ::Float64=0.1,  # stddev of additive Gaussian noise
)
    # 1. Generate design matrix Z ∈ ℝ^{|V|×p}
    Z = randn(num_nodes, p)

    # 2. Generate B ∈ ℝ^p
    # B = rand(Bernoulli(0.5), p)                 # Discrete B
    # if use_continuous_B
    #     β_vec = randn(p)               # Heterogeneous scaling
    #     B = B .* β_vec                          # Element-wise scaling (broadcasted)
    # end
    B = pattern_B

    # 3. Choose c = -0.5 * sqrt(p) to center output
    c = -0.5 * sqrt(p)

    # 4. Additive Gaussian noise ξ ∈ ℝ^{|V|}
    ξ = rand(MvNormal(zeros(num_nodes), σ^2 * I))

    # 5. Generate β ∈ [0,1]^{|V|}
    preact = ((1 / sqrt(p)) .* (Z * B) .+ c) .^ deg .+ ξ
    sigmoid = x -> 1 / (1 + exp(-x))  # Sigmoid function
    β = sigmoid.(preact)

    return Z, β
end

"""
Convert node vector β to edge vector βv.
"""
function β_to_βv(β::Vector, edge_list::DataFrame)

    no_of_edges = size(edge_list, 1)
    βv = zeros(Float64, no_of_edges)

    for e in 1:no_of_edges
        i, j = edge_list[e, :]
        βv[e] = (β[i] + β[j]) / 2.0
    end

    return βv
end

# Example usage
# B = generate_B(10; use_continuous_B=true)
# Z, β = generate_beta(B, 100; p=10, deg=2, σ=0.1)

end # module BetaGen