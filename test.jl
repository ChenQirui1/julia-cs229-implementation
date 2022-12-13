import Pkg 
Pkg.add("SIMD")
using SIMD
using Random

function vadd!(xs::Vector{T}, ys::Vector{T}, ::Type{Vec{N,T}}) where {N, T}
    @assert length(ys) == length(xs)
    @assert length(xs) % N == 0
    lane = VecRange{N}(0)
    @inbounds for i in 1:N:length(xs)
        xs[lane + i] += ys[lane + i]
    end
end

xs = Vector{Float64}(undef,8)
rand!(xs,1:100)

ys = Vector{Float64}(undef,8)
rand!(ys,1:100)

#@code_llvm vadd!(xs, ys, Vec{8,Float64})

#V64 = Vector{Float64}
#@code_llvm axpy!(V64, V64, V64)
#println(V64)

function summation()
    sum = 0
    for i in 1:10
        sum += i
    end
    return sum
end

@code_native summation()