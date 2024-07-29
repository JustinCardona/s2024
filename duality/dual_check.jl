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


function dual_check(s, fn_acc, err_acc, xi_guess::Float64, width::Int64, depth::Int64, depth_true::Int64, samples_domain)
    if depth == 1
        return xi_guess, fn_acc, err_acc[1:depth_true]
    end
    approx = xi -> foldl(+, map(f -> f(xi), fn_acc))
    update(s, 1e-5)
    domain = range(xi_guess - 0.1 * abs(xi_guess), xi_guess + 0.1 * abs(xi_guess), width)
    a = aaa(domain, xi -> eval(s, xi) - approx(xi))
    _, _, z = prz(a)
    try
        z = maximum(filter(x -> abs(a(x)) < 1e-6, map(x -> x.re, z)))
    return dual_check(s, push!(fn_acc, a), push!(err_acc, abs(eval(s, z))), z, width, depth - 1, depth_true, samples_domain)
    catch
        domain = range(xi_guess - 0.1 * abs(xi_guess), xi_guess + 0.1 * abs(xi_guess), 100)
        plot(domain, map(x -> eval(s, x), domain))
        savefig("preview.png")
        println(z)
        readline()
    end
end


# ERROR TESTING
function error_statistics(err_acc, xi_init::Float64, samples::Int64, width::Int64, depth::Int64, reps::Int64)
    if reps == 0
        return foldl(.+, eachcol(err_acc)) / length(err_acc)
    end

    s_init = Surrogate(5)
    domain = range(xi_init - 0.2 * abs(xi_init), xi_init + 0.2 * abs(xi_init), samples)
    a_s = aaa(domain,  xi -> eval(s_init, xi), mmax = Int64(floor(samples / 2)))
    _, _, z = prz(a_s)
    z = maximum(filter(x -> abs(a_s(x)) < 1e-6, map(x -> x.re, z)))
    _, _, err_new = dual_check(s_init, [a_s], [eval(s_init, z)], z, width, depth, depth, domain)
    if isnothing(err_acc)
        return error_statistics(err_new, xi_init, samples, width, depth, reps - 1)
    end
    return error_statistics(hcat(err_acc, err_new), xi_init, samples, width, depth, reps - 1)
end


function err_analysis(xi_init::Float64, hyperparameters, depth::Int64, reps::Int64)
    return map(h -> error_statistics(nothing, xi_init, h[1], h[2], depth, reps), hyperparameters)
end


# xi_init = 1
# samples = 100
# s_init = Surrogate(10)
# domain = range(xi_init - 0.2 * abs(xi_init), xi_init + 0.2 * abs(xi_init), samples)
# a_s = aaa(domain,  xi -> eval(s_init, xi), mmax = Int64(floor(samples / 2)))
# errs = map(xi -> (abs(a_s(xi) - eval(s_init, xi))), domain)
# err_mean = foldl(+, errs) / samples
# _, _, err_new = dual_check(s_init, [a_s], [err_mean], 1.0, 10, 5, 5, domain)
# println(err_new)

samples_domain = 10:20
width_domain = 4:10
depth = 5
reps = Int64(1e1)
hyperparameters = Base.product(samples_domain, width_domain)
errs = err_analysis(1.0, hyperparameters, depth, reps)
serialize("hyperparameters.dat", hyperparameters)
serialize("errs.dat", errs)

n = length(hyperparameters)
s = reshape(map(h -> h[1], hyperparameters), n)
w = reshape(map(h -> h[2], hyperparameters), n)
e = reshape(map(e -> foldl(+, e) / depth, errs), n)
surface(s, w, e)
xlabel!("samples")
ylabel!("width")
savefig("preview.png")