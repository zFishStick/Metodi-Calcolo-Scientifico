# Decomposizione di Cholesky in Julia
using LinearAlgebra
using SparseArrays, MAT


function cholesky_decomposition()

    #folder = "C:\\Users\\Diagon\\Desktop\\UNIMIB\\ANNO 1\\SECONDO SEMESTRE\\Metodi Calcolo\\Matrici-Sparse"
    folder = "matrici\\"
    files = readdir(folder)
    matrici = []
    for f in files
        data = matread(joinpath(folder, f))
        push!(matrici, (data["Problem"]["A"], data["Problem"]["name"]))
    end

    #ordino le matrici in base alla dimensione
    matrici = sort(matrici, by = m -> m[1].n)
    println([(t[2], t[1].n) for t in matrici])

    for (A, A_name) in matrici
        # Controllo se la matrice è simmetrica
        if !(issymmetric(A))
            error("La matrice deve essere simmetrica.")
        end

        mem_prima = Base.gc_live_bytes() / 1024^2

        println("Matrice: ", A_name)
        # Controllo se la matrice è definita positiva, inutile perchè la sua descrizione è
        # "Test whether a matrix is positive definite (and Hermitian) by trying to perform a Cholesky factorization of A."
        # if !(isposdef(A))
        #     error("La matrice deve essere definita positiva.")
        # end

        # Soluzione esatta xe = [1,1,1,1...]
        xe = ones(size(A, 1))
        
        b = A * xe
        t = @elapsed begin

            f = cholesky(A)
            x = f \ b
             mem_dopo = Base.gc_live_bytes() / 1024^2

        end

        

        println("errore: ", norm(x-xe)/norm(xe))
        println("tempo di esecuzione: ", t, " s")
        println("aumento di memoria: ", mem_dopo - mem_prima, " MB")

    end

    return
end

cholesky_decomposition() 
