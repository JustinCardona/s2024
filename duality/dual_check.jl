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


function dual_check(fn_acc, xi_guess::Float64, depth::Int64)
    if depth == 0
        return xi_guess, fn_acc
    else
        approx = xi -> foldl(+, map(f -> f(xi), fn_acc))
        update(s, 1e-5)
        a = aaa(range(0.9 * xi_guess, 1.1 * xi_guess, 4), xi -> eval(s, xi) - approx(xi))
        _, _, z = prz(a)
        return dual_check(push!(fn_acc, a), z[1].re, depth-1)
    end
end


xi_init = 1
s = Surrogate(10)
domain = range(0.0 * xi_init, 1.2 * xi_init, 10)
a_s = aaa(domain,  xi -> eval(s, xi))
_, _, z = prz(a_s)
println(dual_check([a_s], z[1].re, 10)[1])