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
    a = aaa(range(0.8 * xi_guess, 1.2 * xi_guess, width), xi -> eval(s, xi) - approx(xi))
    _, _, z = prz(a)
    return dual_check(s, push!(fn_acc, a), z[1].re, depth-1, width)
end


# PERFORMANCE TESTING
function error_analysis(xi_init::Float64, w_init::Int64, w::Int64, d::Int64)
    n_samples = 100
    s = Surrogate(10)
    domain = range(0.8 * xi_init, 1.2 * xi_init, w_init)

    a_s = aaa(domain,  xi -> eval(s, xi))
    xi_new, fn_acc = dual_check(s, [a_s], xi_init, d, w)
    approx = xi -> foldl(+, map(f -> f(xi), fn_acc))    
    domain = range(0.95 * xi_new, 1.05 * xi_new, n_samples)
    err = map(xi -> (abs(approx(xi) - eval(s, xi))), domain)
    err_mean = foldl(+, err) / n_samples
    # plot(domain, err)
    # savefig("err.png")
    # println(w, " ", d, " ", err_mean)
    # readline()
    return err_mean
end


width_init = [100]
width = range(4, 6)
depth = range(0, 2)
hyperparameters = Base.product(width_init, width, depth)

ws = []
ds = []
errs = []
stds = []

n_reps = 1e3
i = 1
for h in hyperparameters
    try
        err = map(none -> error_analysis(1.0, h[1], h[2], h[3]), range(0, n_reps))
        err_mean = foldl(+, err) / n_reps
        err_std = sqrt(foldl(+, map(x -> (x - err_mean)^2, err))) / n_reps
        append!(ws, h[2])
        append!(ds, h[3])
        append!(errs, err_mean)
        append!(stds, err_std)
    catch
        println("what")
    end    
    global i += 1
end

scatter(ws, ds, errs)
xlabel!("width")
ylabel!("depth")
savefig("preview.png")