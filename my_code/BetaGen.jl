module BetaGen

using Random, LinearAlgebra, Distributions

export generate_B, generate_beta

function generate_B(
    p::Int;           # feature dimension
    use_continuous_B::Bool = false
)
    B = rand(Bernoulli(0.5), p)                 # Discrete B
    if use_continuous_B
        β_vec = randn(p)               # Heterogeneous scaling
        B = B .* β_vec                          # Element-wise scaling (broadcasted)
    end

    return B
end

function generate_beta(
    pattern_B::Vector, # underlying pattern (matrix B)
    num_nodes::Int;    # |V|
    p::Int = 10,       # feature dimension
    deg::Int = 2,      # non-linearity degree
    σ::Float64 = 0.1,  # stddev of additive Gaussian noise
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

# Example usage
B = generate_B(10; use_continuous_B=true)
Z, β = generate_beta(B, 100; p=10, deg=2, σ=0.1)

end # module BetaGen