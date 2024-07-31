using LinearAlgebra, BaryRational, Serialization, Plots


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
        P0 = (A + Adjoint(A)) / 2;
        P1 = rand(ComplexF64, n, n);
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


a::BaryRational.AAAapprox + b::BaryRational.AAAapprox = BaryRational.AAAapprox(push(a.z, b.z), push(a.f, b.f), push(a.w, b.w), push(a.errvec, b.errvec))


s = Surrogate(100)
domain = range(0, 10, 20)
approx = aaa(domain, x -> eval(s, x))
println(typeof(approx))

for i in range(1, 10)
    approx = approx + err
    plot(domain, [map(x -> eval(s, x), domain), map(approx, domain)])
    savefig("preview.png")
    readline()
    update(s, 1e-3)
end