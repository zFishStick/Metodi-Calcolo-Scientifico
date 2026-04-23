# Decomposizione di Cholesky in Julia
using LinearAlgebra
using SparseArrays, MAT
using CSV, DataFrames

function cholesky_decomposition()

    if Sys.iswindows()
        folder = "C:\\Users\\Diagon\\Desktop\\UNIMIB\\ANNO 1\\SECONDO SEMESTRE\\Metodi Calcolo\\Matrici-Sparse";
    else
        folder = "/home/diagon/Matrici-Sparse";
    end
    #folder = "matrici\\"
    files = readdir(folder)
    matrici = []
    dimensioni = []
    errori = []
    tempi = []
    memorie = []
    for f in files
        data = matread(joinpath(folder, f))
        if !Sys.iswindows() && f == "Flan_1565.mat" && false
            println("Salto $f: troppo grande per la RAM disponibile (26GB/16GB swap).")
            continue
        else
            push!(matrici, (data["Problem"]["A"], data["Problem"]["name"]))
        end
        
    end


    results = DataFrame(
        nome = String[],
        dimensione = Int[],
        tempo = Float64[],
        errore = Float64[],
        memoria = Float64[]
    )

    #ordino le matrici in base alla dimensione
    matrici = sort(matrici, by = m -> m[1].n)
    dimensioni = [m[1].n for m in matrici]
    println(dimensioni)
    println([(t[2], t[1].n) for t in matrici])

    for (A, A_name) in matrici
        # Controllo se la matrice è simmetrica
        if !(issymmetric(A))
            error("La matrice deve essere simmetrica.")
        end


        println("Matrice: ", A_name)
        # Controllo se la matrice è definita positiva, inutile perchè la sua descrizione è
        # "Test whether a matrix is positive definite (and Hermitian) by trying to perform a Cholesky factorization of A."
        # if !(isposdef(A))
        #     error("La matrice deve essere definita positiva.")
        # end

        # Soluzione esatta xe = [1,1,1,1...]
        xe = ones(size(A, 1))
        mem_prima = Base.summarysize(A) / 1024^2
        println("mem_prima :", mem_prima)
        b = A * xe
        t = @elapsed begin
            
            f = cholesky(A)
            x = f \ b

        end

        s = unsafe_load(f.ptr)

        mem_stimata = (
            (s.xsize * sizeof(Float64) + 
            s.ssize * sizeof(Int64)+ 
            s.nsuper * sizeof(Int64) * 3 +
            length(x) * sizeof(Int64)) / 1024^2
        )

        err = norm(x-xe)/norm(xe)
        #memoria_allocata = mem_stimata - mem_prima
        #push!(errori, err)
        #push!(tempi, t)
        #push!(memorie, memoria_allocata)
        
        push!(results, (
            A_name,
            size(A,1),
            t,
            err,
            mem_stimata - mem_prima
        ))
   end
    filename = Sys.iswindows() ? "risultati_win_julia.csv" : "risultati_linux_julia.csv"
                             
    CSV.write(filename, results)

    return
end

cholesky_decomposition() 
