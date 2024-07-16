using LinearAlgebra, BaryRational, Plots


mutable struct Surrogate
    a0::Float64
    a1::Float64
    s::Vector{ComplexF64}
    P0::Matrix{ComplexF64}
    P1::Matrix{ComplexF64}

    function Surrogate(n::Int64)
        a0 = rand(Float64)
        a1 = rand(Float64)
        s = rand(ComplexF64, n)
        A = rand(ComplexF64, n, n);
        P0 = rand(ComplexF64, n, n);
        P1 = (A + Adjoint(A)) / 2;
        new(a0, a1, s, P0, P1)
    end
end


function eval(surr::Surrogate, xi::Float64)
    M = (surr.a1 + xi) .* surr.P0 + surr.a0 .* surr.P1;
    x = M \ surr.s;
    return ((Adjoint(x) * surr.s).im - Adjoint(x) * surr.P0 * x).re    
end


function update(surr::Surrogate, dx::Float64)
    n = size(surr.s)[1]
    A = dx * rand(ComplexF64, n, n);
    surr.P0 += dx * rand(ComplexF64, n, n);
    surr.P1 += (A + Adjoint(A)) / 2;
end


function dual_check(s, fn_acc, xi_guess::Float64, depth::Int64, width::Int64)
    if depth == 0
        return xi_guess, fn_acc
    end
    approx = xi -> foldl(+, map(f -> f(xi), fn_acc))
    update(s, 1e-5)
    a = aaa(range(0.9 * xi_guess, 1.1 * xi_guess, width), xi -> eval(s, xi) - approx(xi))
    _, _, z = prz(a)
    try
        return dual_check(s, push!(fn_acc, a), z[1].re, depth-1, width)
    catch
        return dual_check(s, push!(fn_acc, a), xi_guess, depth-1, width)
    end
end


# PERFORMANCE TESTING
function error_analysis(w_init::Int64, w::Int64, d::Int64)
    xi_init = 1
    n_samples = 1000

    s = Surrogate(10)
    domain = range(0.8 * xi_init, 1.2 * xi_init, w_init)
    a_s = aaa(domain,  xi -> eval(s, xi))
    _, _, z = prz(a_s)
    xi_new, fn_acc = dual_check(s, [a_s], z[1].re, d, w)
    approx = xi -> foldl(+, map(f -> f(xi), fn_acc))    
    domain = range(0.8 * xi_new, 1.2 * xi_new, n_samples)
    err = map(xi -> approx(xi) - eval(s, xi), domain)
    err_mean = foldl(+, err) / n_samples
    err_std = sqrt(foldl(+, map(x -> (x - err_mean)^2, err))) / n_samples
    return err_mean, err_std
end


width_init = [15]
width = range(4, 10)
depth = range(0, 10)
hyperparameters = Base.product(width_init, width, depth)
errs = zeros(length(hyperparameters), 4)

n_reps = 100
i = 1
for h in hyperparameters
    err = map(none -> error_analysis(h[1], h[2], h[3]), range(0, n_reps))
    err_mean = foldl(.+, err) ./ n_reps
    err_var = foldl(.+, map(x -> (x .- err_mean).^2, err))
    err_std = sqrt(err_var[1]), sqrt(err_var[2]) ./ n_reps
    errs[i, :] = [err_mean[1], err_mean[2], err_std[1], err_std[2]]
    global i += 1
end

surface(width, depth, errs[:, 1])
savefig("preview.png")