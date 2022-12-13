using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("BenchmarkTools")
using CSV
using DataFrames
using Random
using BenchmarkTools
using LinearAlgebra

df = CSV.read("..//dataset//Iris.csv", DataFrame)

#Declaring global varib
const X = hcat(ones(150),Matrix(df[:,2:4])) #design matrix
θ = Vector{Float64}(undef,size(X)[2])
rand!(θ,1:100)
const y = df[:,5]

#hypothesis function
h(x) = θ'x

hdot(x) = x ⋅ θ

hsum(x) = sum(x .* θ)
    
function hloop(x)
    sum = 0
    for (idx,xi) in enumerate(x)
        sum += xi*θ[idx]
    end
    return sum
end

# @btime h($X[1,:])
# @btime hloop($X[1,:])
# @btime hsum($X[1,:])
# @btime hdot($X[1,:])


# Jsum(θ::Vector) = 0.5*sum((X*θ-y).*(X*θ-y))

# Jdot(θ::Vector) = 0.5*dot((X*θ-y),(X*θ-y))
# @btime Jsum($θ)
# @btime Jdot($θ)
transX = transpose(X)

#sum(X,dims=3)
function BatchGradientDescent(X,y,epochs,α)
    m,n = size(X) #3 X 150
    for epoch in 1:epochs
        #println(epoch)
        for j in 1:n
            batchsum = 0
            for i in 1:m
                #batchsum += (y[row_index] - h(x))*x[i,j]
                batchsum += (y[i] - hdot(X[i,:]))*X[i,j]
            end;
            #println("batchsum: " ,batchsum)
            #println(batchsum)
            θ[n] += α*batchsum
        end;
    end;
    return θ
    end;


function BatchGradientDescentOld(X,y,epochs,α)
    for epoch in 1:epochs
        for (idxθj,θj) in enumerate(θ)
            batchsum = 0
            for (row_index, x) in enumerate(eachrow(X))
                batchsum += (y[row_index] - hdot(x))*x[idxθj]
            end;
            θj += α*batchsum
            θ[idxθj] = θj
        end;
    end;
    return θ
    end;


#@btime X[2,4]


# @btime batch = Vector{Float64}(undef,10000)

# batch[1] = 0.3
# batch[2] = 0.4

# batch


# function summation(m)
#     sum = 0
#     for i in 1:m
#         sum += i
#     end
#     return sum   
# end

# function summationArray(m)
#     sumArray = Vector{Float64}(undef,m)
#     for i in 1:m
#         sumArray[i] = i
#     end
#     return sum(sumArray)
# end

# @btime summation(100000)

# @btime summationArray(100000)