# Decomposizione di Cholesky in Julia
using LinearAlgebra
using SparseArrays, MAT
using Plots

function cholesky_decomposition()

    #folder = "C:\\Users\\Diagon\\Desktop\\UNIMIB\\ANNO 1\\SECONDO SEMESTRE\\Metodi Calcolo\\Matrici-Sparse"
    folder = "matrici\\"
    files = readdir(folder)
    matrici = []
    dimensioni = []
    errori = []
    tempi = []
    memorie = []
    for f in files
        data = matread(joinpath(folder, f))
        push!(matrici, (data["Problem"]["A"], data["Problem"]["name"]))
    end

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
        #println(fieldnames(typeof(s)))

        mem_stimata = (
            (s.xsize * sizeof(Float64) + 
            s.ssize * sizeof(Int64)+ 
            s.nsuper * sizeof(Int64) * 3 +
            length(x) * sizeof(Int64)) / 1024^2
        )

        errore = norm(x-xe)/norm(xe)
        memoria_allocata = mem_stimata - mem_prima
        push!(errori, errore)
        push!(tempi, t)
        push!(memorie, memoria_allocata)
        println("Peso stimato:   ", round(mem_stimata, digits=3), " MB")
        println("errore: ", errore)
        println("tempo di esecuzione: ", t, " s")
        println("Memoria allocata: ", memoria_allocata, " MB")

       
   end

    
    p1 = plot(dimensioni, errori,
        xlabel = "Dimensione matrice",
        ylabel = "Errore relativo",
        title  = "Errore vs Dimensione",
        xscale = :log10,
        yscale = :log10,
        marker = :circle,
        lw     = 2,
        label  = "errore")

    p2 = plot(dimensioni, tempi,
        xlabel = "Dimensione matrice",
        ylabel = "Tempo (s)",
        title  = "Tempo vs Dimensione",
        xscale = :log10,
        marker = :circle,
        lw     = 2,
        label  = "tempo")

    p3 = plot(dimensioni, memorie,
        xlabel = "Dimensione matrice",
        ylabel = "Memoria (MB)",
        title  = "Memoria vs Dimensione",
        xscale = :log10,
        marker = :circle,
        lw     = 2,
        label  = "memoria")

    # Tutti e tre affiancati in un unico file
    plot(p1, p2, p3, layout=(1,3), size=(1200,400))
    savefig("risultati.png")
    return
end

cholesky_decomposition() 
