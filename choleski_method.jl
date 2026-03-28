# Decomposizione di Cholesky in Julia
using LinearAlgebra


function cholesky_decomposition(A::Matrix{Float64})

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

    f = cholesky(A)

    println("Fattorizzazione di Cholesky: ", f)

    x = f \ b

    println("Soluzione: ", x)

    return x
        
end

A = [4.0 2.0; 2.0 3.0]
L = cholesky_decomposition(A)
