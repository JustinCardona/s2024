using LinearAlgebra, BaryRational, Serialization, Plots


mutable struct Surrogate
    a0::Float64
    a1::Float64
    s::AbstractVector{ComplexF64}
    P0::AbstractMatrix{ComplexF64}
    P1::AbstractMatrix{ComplexF64}

    function Surrogate(n::Int)
        a0 = rand(Float64) .- 0.5
        a1 = rand(Float64) .- 0.5
        s = rand(ComplexF64, n) .- 0.5(1 + 1im)
        A = rand(ComplexF64, n, n) .- 0.5(1 + 1im)
        P0 = (A + adjoint(A)) / 2;
        P1 = rand(ComplexF64, n, n) .- 0.5(1 + 1im)
        new(a0, a1, s, P0, P1)
    end
end


function eval(surr::Surrogate, xi::Float64)
    M = (surr.a0 + xi) .* surr.P0 + surr.a1 .* surr.P1
    x = M \ surr.s
    return real(imag(adjoint(x) * surr.s) - adjoint(x) * surr.P0 * x)
end


function update(surr::Surrogate, dx::Float64)
    n = size(surr.s)[1]
    A = dx * (rand(ComplexF64, n, n) .- 0.5(1 + 1im))
    surr.P0 += dx * (rand(ComplexF64, n, n) .- 0.5(1 + 1im))
    surr.P1 += (A + adjoint(A)) / 2;
end


function approximate(s::Surrogate, z::Float64, width::Int, domain::AbstractVector{Float64}, codomain::AbstractVector{Float64}, tol=1e-3)
    domain_z = range(z - tol * abs(z), z + tol * abs(z), width)
    domain_new = vcat(domain, domain_z)
    codomain_new = vcat(codomain, map(x -> eval(s, x), domain_z))
    return aaa(domain_new, codomain_new, clean=1), domain_new, codomain_new
end


function find_zero(s::Surrogate, z::Float64, width_init::Int, width::Int, domain::AbstractVector{Float64}, codomain::AbstractVector{Float64}, tol=1e-3, search_depth=0)
    a, domain, codomain = approximate(s, z, width_init, domain, codomain, tol)
    _, _, zeros = prz(a)
    z = maximum(map(x -> real(x), zeros))
    if abs(eval(s, z)) < tol || search_depth == 15
        return z, a, domain, codomain
    end
    return find_zero(s, z, width, width, domain, codomain, tol, search_depth+1)
end


function optimize_surr(s::Surrogate, z::Float64, width_init::Int, width::Int, depth::Int, tol=1e-4, dx=1e-2)
    z, a, domain, codomain = find_zero(s, z, width_init, width, Array{Float64}(undef, 0), Array{Float64}(undef, 0), tol)
    errs = Array{Float32}(undef, depth, 3)
    for i in range(1, depth)
        z, a, domain, codomain = find_zero(s, z, width, width, domain, codomain, tol)
        errs[i, :] = [z, abs(eval(s, z)), length(domain)]
        update(s, dx)
    end
    return errs
end

# TESTING

samples = 1e6
depth = 5
s_size = 100
# n, errs = 0, zeros(depth, 3)
n, errs = deserialize("errs_good.dat")
@Threads.threads for i in range(1, samples)
    try
        global errs += optimize_surr(Surrogate(s_size), 1.0, 8, 2, depth, 1e-2, 1e-2)
        global n += 1
    catch
        println("ERROR")
    end
    serialize("errs_good.dat", [n, errs])
    println(n)
end
errs /= n
samples = vcat(errs'[3, 1], map(i -> errs'[3, i] - errs'[3, i-1], range(2, size(errs, 1))))
println(errs'[1, :])
println(errs'[2, :])
println(samples)
