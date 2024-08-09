using Serialization

e = deserialize("errs.dat")
d = size(e)
# errs = foldl(.+, e) / (d[1] * d[2])

println(sum(e, dims = 1))