using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("ForwardDiff")
using CSV
using DataFrames
using ForwardDiff
using Random
using LinearAlgebra

df = CSV.read("..//dataset//Iris.csv", DataFrame)

#Declaring global varib
const X = hcat(ones(150),Matrix(df[:,2:4])) #design matrix
θ = Vector{Float64}(undef,size(X)[2])
rand!(θ,1:100)
const y = df[:,5]

#hypothesis function
h(x) = dot(θ,x)

#cost function J
#J(θ::Vector) = (X*θ-y)'(X*θ-y)

J(θ::Vector) = 0.5*sum((X*θ-y).*(X*θ-y))


function BatchGradientDescent(X,y,epochs,α)
    for epoch in 1:epochs
        #println(epoch)
        for (idxθj,θj) in enumerate(θ)
            batchsum = 0
            for (row_index, x) in enumerate(eachrow(X))
                batchsum += (y[row_index] - h(x))*x[idxθj]
            end;
            #println("batchsum: " ,batchsum)
            θj += α*batchsum
            θ[idxθj] = θj
        end;
        #println("Cost Func:", J(θ))
    end;
    return θ
    end;

MinMaxNorm(x,Xmin,Xmax) = (x - Xmin) / (Xmax- Xmin)


