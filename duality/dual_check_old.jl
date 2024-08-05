using LinearAlgebra, BaryRational, Serialization, Plots
import Base.:+

a::BaryRational.AAAapprox + b::BaryRational.AAAapprox = BaryRational.AAAapprox(vcat(a.x, b.x), vcat(a.f, b.f), vcat(a.w, b.w), vcat(a.errvec, b.errvec))


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
    # M = (surr.a1 + xi) .* surr.P0 + surr.a0 .* surr.P1;
    M = (surr.a1 + xi) .* surr.P1;# + surr.a0 .* surr.P1;
    x = M \ surr.s;
    return ((Adjoint(x) * surr.s).im - Adjoint(x) * surr.P0 * x).re    
end


function update(surr::Surrogate, dx::Float64)
    n = size(surr.s)[1]
    A = dx * rand(ComplexF64, n, n);
    surr.P0 += dx * rand(ComplexF64, n, n);
    surr.P1 += (A + Adjoint(A)) / 2;
end


function dual_check(s::Surrogate, approx::BaryRational.AAAapprox, err_acc, xi_guess::Float64, width::Int64, depth::Int64, tol = 1e-3)
    if depth == 1
        return approx, err_acc, xi_guess
    end
    _, _, z = prz(approx)
    z = maximum(map(x -> x.re, filter(x -> x.im < tol, z)))
    update(s, 1e-2)
    err = aaa(range(z - 0.1 * abs(z), z + 0.1 * abs(z), 5), x -> approx(x) - eval(s, x))
    return dual_check(s, approx + err, push!(err_acc, abs(eval(s, z))), z, width, depth - 1)
end


# ERROR TESTING
function error_statistics(err_acc, xi_init::Float64, samples::Int64, width::Int64, depth::Int64, reps::Int64)
    if reps == 0
        return foldl(.+, eachcol(err_acc)) / length(err_acc)
    end

    s_init = Surrogate(2)
    domain = range(xi_init - 0.2 * abs(xi_init), xi_init + 0.2 * abs(xi_init), samples)
    a_s = aaa(domain,  xi -> eval(s_init, xi), mmax = Int64(floor(samples / 2)))
    _, _, z = prz(a_s)
    z = maximum(filter(x -> abs(a_s(x)) < 1e-6, map(x -> x.re, z)))

    _, err_new, _ = dual_check(s_init, a_s, [abs(eval(s_init, z))], z, width, depth)
    if isnothing(err_acc)
        return error_statistics(err_new, xi_init, samples, width, depth, reps - 1)
    end
    return error_statistics(hcat(err_acc, err_new), xi_init, samples, width, depth, reps - 1)
end


function err_analysis(xi_init::Float64, hyperparameters, depth::Int64, reps::Int64)
    return map(h -> error_statistics(nothing, xi_init, h[1], h[2], depth, reps), hyperparameters)
end


samples_domain = 10:20
width_domain = 4:10
depth = 5
reps = Int64(1e2)
hyperparameters = Base.product(samples_domain, width_domain)
@time errs = err_analysis(1.0, hyperparameters, depth, reps)
serialize("hyperparameters.dat", hyperparameters)
serialize("errs.dat", errs)

n = length(hyperparameters)
s = reshape(map(h -> h[1], hyperparameters), n)
w = reshape(map(h -> h[2], hyperparameters), n)
e = reshape(map(e -> foldl(+, e) / depth, errs), n)
scatter(s, w, e)
xlabel!("samples")
ylabel!("width")
savefig("preview.png")