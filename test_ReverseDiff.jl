"""Testing automatic differentiation package (ForwardDiff or ReverseDiff)
on a small size neural networks:
y = ∑_i w[i] * x[i]
z = tanh(y),
where We would like to compute the gradient ∂z/∂w[i].

Exact solution by chain rule:
∂z/∂w[i] = ∂z/∂y * ∂y/∂w[i]
= sech(y)^2 * x[i]
= sech( sum( w .* x) )^2 * x[i].

ReverseDiff is more efficient than ForwardDiff when output dimension is less than
the input dimension. Therefore ReverseDiff is a better choice for computing gradient.

However, the documentation of ReverseDiff is less detailed for the moment.
One can go to the website of ForwardDiff for some general information.

-- Bo Li
"""

# using ForwardDiff
using ReverseDiff


function f1(w, x)
    """Input:
    x -> a vector of size N,
    w -> a vector of size N (tunable weight parameters),
    Ouput:
    y -> a scalar variable.
    """
    y = sum( w .* x )
    return y
end


function f2(y)
    """
    Output:
    z -> a scalar variable.
    """
    z = tanh(y)
    return z
end


x0 = [1., -0.5]
w0 = [2., 3.]

output = w -> f2(f1(w, x0))             ## fix x as x0, view w as decision variables
g = ReverseDiff.gradient(output, w0)    ## the gradient of output function evaluated at w0

g_true = [sech( sum( w0 .* x0) )^2 *x0[i] for i in 1:2]    ## exact gradient

println("gradient by autodiff:  $(g)")
println("exact gradient:        $(g_true) ")

## Comment: results for the above parameter setting:
# gradient by autodiff:  [0.786448, -0.393224]
# exact gradient:        [0.786448, -0.393224]
