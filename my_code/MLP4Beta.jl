module MLP4Beta

using Flux

export train_mlp

# Define an MLP model: input dim p, one hidden layer with 32 units, output dim 1
function build_mlp(p::Int; hidden_units=32)
    return Chain(
        Dense(p, hidden_units, relu),   # hidden layer with ReLU
        Dense(hidden_units, 1)          # output layer
    )
end

# Fit the MLP to data Z (n x p) and β (n)
"""
    train_mlp(Z::Array{Float32, 3}, β::Array{Float32, 2}; epochs=100, lr=0.01) -> Chain

Trains a Multi-Layer Perceptron (MLP) model to fit the input data `Z` and target values `β`.

# Arguments
- `Z::Array{Float32, 3}`: A 3D array of shape `(n, p, instance_num)` where `n` is the number of samples, 
  `p` is the number of features, and `instance_num` is the number of instances.
- `β::Array{Float32, 2}`: A 2D array of shape `(n, instance_num)` where `n` is the number of samples 
  and `instance_num` is the number of instances.
- `epochs::Int` (optional): The number of training epochs. Default is `100`.
- `lr::Float64` (optional): The learning rate for the Adam optimizer. Default is `0.01`.

# Returns
- `Chain`: The trained MLP model.

# Details
The function builds an MLP model using the `build_mlp` function (assumed to be defined elsewhere). 
It uses the Mean Squared Error (MSE) as the loss function and the Adam optimizer for training. 
The training process involves iterating over the specified number of epochs and updating the model 
parameters based on the computed gradients for each instance in the dataset.
"""
function train_mlp(Z::Array{Float32, 3}, β::Array{Float32, 2}; epochs=100, lr=0.01)
    n, p, instance_num = size(Z)
    
    # Build the model
    model = build_mlp(p)
    
    # Define loss function (Mean Squared Error)
    loss_fn = Flux.Losses.mse

    # Define optimizer
    opt_state = Flux.setup(Adam(lr), model)

    losses = Float64[]

    # Training loop
    for epoch in 1:epochs
        total_loss = 0.0
        
        for i in 1:instance_num
            # Grab the i-th instance
            Zi = Z[:, :, i]    # (n, p)
            βi = β[:, i]       # (n,)
            
            # Prepare input: transpose to (p, n)
            input = Zi'
            
            # Target: reshape βi to (1, n)
            target = reshape(βi, 1, :)
            
            # Compute and apply gradients
            loss, grads = Flux.withgradient(model) do m
                y_pred = m(input)
                loss_fn(y_pred, target)
            end

            Flux.update!(opt_state, model, grads[1])
            # Accumulate loss for monitoring
            total_loss += loss
        end

        avg_loss = total_loss / instance_num
        losses = push!(losses, avg_loss)
        # Print loss every 10 epochs
        if epoch % 10 == 0
            println("Epoch $epoch: Avg Loss = ", avg_loss)
        end
    end
    
    return model
end

# # Example Usage

# # Example data
# n, p, instance_num = 50, 10, 100
# Z = randn(Float32, (n, p, instance_num))
# true_w = randn(Float32, p)
# β = [Z[:, :, i] * true_w + 0.1f0 * randn(Float32, n) for i in 1:instance_num]
# β = hcat(β...)  # (n, instance_num)

# # Train
# model = train_mlp(Z, β; epochs=100, lr=0.01)

# # Predict
# Z_test = randn(Float32, 5, p)
# β_pred = model(Z_test')  # output will be of size (1, 5)
# println("Predictions: ", β_pred)

end # module MLP4Beta