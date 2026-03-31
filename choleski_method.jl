# Decomposizione di Cholesky in Julia
using LinearAlgebra
using SparseArrays, MAT


function cholesky_decomposition(A::Matrix{Float64}, b::Vector{Float64} = Float64[])

    folder = "matrici\\"
    files = readdir(folder)
    matrici = []
    for f in files
        data = matread(joinpath(folder, f))
        push!(matrici, data["Problem"]["A"])
    end

    for A in matrici
        # Controllo se la matrice è simmetrica
        if !(issymmetric(A))
            error("La matrice deve essere simmetrica.")
        end

        # Controllo se la matrice è definita positiva, inutile perchè la sua descrizione è
        # "Test whether a matrix is positive definite (and Hermitian) by trying to perform a Cholesky factorization of A."
        # if !(isposdef(A))
        #     error("La matrice deve essere definita positiva.")
        # end

        # Soluzione esatta xe = [1,1,1,1...]
        xe = ones(size(A, 1))
        
        b = A * xe
       
        x = A \ b

        println("errore: ", norm(x-xe))/norm(xe)

    end

    return 
end

A = [4.0 10.0 8.0; 10.0 26.0 26.0; 8.0 26.0 61.0]

cholesky_decomposition(A, b) 
