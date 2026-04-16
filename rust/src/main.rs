
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::{Mat, Side};
use sprs::io::read_matrix_market;
use cap::Cap;
use std::alloc::System;
use std::fs::File;
use csv::Writer;

use faer::sparse::linalg::cholesky;
use faer::sparse::linalg::cholesky::{CholeskySymbolicParams, SymmetricOrdering};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::Par;
use faer::linalg::cholesky::llt::factor::LltRegularization;
use faer::Conj;

#[global_allocator]
static ALLOCATOR: Cap<System> = Cap::new(System, usize::MAX);

fn main() {
    let folder = "C://Users//gabri//OneDrive//Desktop//matrici";

    // let folder = "C:\\Users\\Simone\\Desktop\\Università\\Magistrale\\Metodi del calcolo scientifico\\Progetto1\\matrici_mtx";

    // let matrix_list = [
    //     "Flan_1565", "StocF-1465", "cfd2", "cfd1", "G3_circuit",
    //     "parabolic_fem", "apache2", "shallow_water1", "ex15",
    // ];

    let matrix_list = ["apache2", "ex15", "cfd2", "cfd1", "parabolic_fem", "shallow_water1", "G3_circuit", "Flan_1565", "StocF-1465"];
    
    for name in matrix_list {
        println!("\n--- Analisi Matrice: {} ---", name);
        let path: &str = &format!("{}/{}.mtx", folder, name);

        let matrix_sprs = read_matrix_market::<f64, usize, _>(path).expect("Errore nella lettura");
        
        let matrix_csc = matrix_sprs.to_csc::<usize>();
        
        let (nrows, ncols) = (matrix_csc.rows(), matrix_csc.cols());

        let (indptr, indices, data) = matrix_csc.into_raw_storage();

        let symbolic_mat = SymbolicSparseColMat::<usize>::new_checked(
            nrows, ncols,
            indptr.to_vec(),
            None,
            indices.to_vec(),
        );
        
        let matrix_faer = SparseColMat::new(symbolic_mat.clone(), data);

        let xe = Mat::<f64>::from_fn(ncols, 1, |_, _| 1.0);
        let mut b = &matrix_faer * &xe;
        
        let mem_prima = ALLOCATOR.allocated();
        let time = std::time::Instant::now();
        //fino a qui tutto uguale a prima

        //uso un'altro modulo della libreria, uso faer...::cholesky al posto di ::solve
        //calcolo matrice simbolica della fattorizzazione cholesky
        let symbolic_cholesky = cholesky::factorize_symbolic_cholesky(
            symbolic_mat.as_ref(),
            Side::Lower,
            SymmetricOrdering::Amd, // parametri di default
            CholeskySymbolicParams::default(), //parametri di default
        ).expect("errore in simbolyc_cholesky");

        //dopo aver calcolato la matrice simbolica devo calcolare i valori per riempirla
        
        //per qualche motivo la funzione per calcolare i valori ha bisogno di un buffer per funzionare
        let mut l_values = vec![0.0f64; symbolic_cholesky.len_val()];

        // parametri di default della fase numerica
        let par = Par::Seq;
        let regularization = LltRegularization::default();
        let llt_params = Default::default();

        //Per funzionare ha bisogno di uno stack di memoria (penso per farci le operazioni sopra bho)
        let req = symbolic_cholesky.factorize_numeric_llt_scratch(par, llt_params); //funzione che calcola la memoria dello stack
        let mut stack_buf = MemBuffer::new(req);
        let mut stack = MemStack::new(&mut stack_buf);

        //funzione che calcola effettivamente i valori della matrice
        let factor = symbolic_cholesky.factorize_numeric_llt(
            &mut l_values, //buffer di prima
            matrix_faer.as_ref(), //matrice originale
            Side::Lower, //parametri di default
            regularization, //parametri di default
            par, //parametri di default
            &mut stack, //stack di prima 
            llt_params, //parametri di default
        ).expect("errore in numeric_factorize");
        
        //ora abbiamo la matrice fattorizzata, passiamo alla risoluzione del sistema

        //per risolvere il sistema serve, come prima, anche per la risoluzione uno stack di memoria
        let req = symbolic_cholesky.solve_in_place_scratch::<f64>(1, Par::Seq); //funzione che calcola la memoria per la risoluzione
        let mut stack_buf = MemBuffer::new(req);
        let mut stack = MemStack::new(&mut stack_buf);

        factor.solve_in_place_with_conj(
            Conj::No, //riguarda matrici di numeri complesso quindi mettiamo NO
            b.as_mut(), //vettore soluzione, viene modificato in place e diventerà il nostro vettore x
            Par::Seq, //parametri di default
            &mut stack, //stack di prima
        );

        let x = b; //rinomino per comodità
        
        //da qui in poi tutto uguale a prima
        let elapsed = time.elapsed();
        let mem_dopo = ALLOCATOR.allocated();

        let diff = &x - &xe;
        let rel_error = diff.norm_l2() / xe.norm_l2();
        let aumento_memoria:f64 = mem_dopo.saturating_sub(mem_prima) as f64;

        println!("Memoria allocata: {} MB", aumento_memoria / (1024.0 * 1024.0));
        println!("Errore relativo: {:e}", rel_error);
        println!("Tempo di esecuzione: {:.2?}", elapsed);

        write_results_csv(
            "risultati_rust.csv",
            name,
            matrix_sprs.cols(),
            elapsed.as_secs_f64(),
            rel_error,
            aumento_memoria
        );

    }

}

fn write_results_csv(
    filename: &str,
    nome: &str,
    dimensione: usize,
    tempo: f64,
    errore: f64,
    memoria: f64
) {
    let file_exists = std::path::Path::new(filename).exists();

    let file = File::options()
        .append(true)
        .create(true)
        .open(filename)
        .unwrap();

    let mut wtr = Writer::from_writer(file);

    if !file_exists {
        wtr.write_record(&["nome", "dimensione", "tempo", "errore", "memoria"])
            .unwrap();
    }

    wtr.write_record(&[
        nome,
        &dimensione.to_string(),
        &tempo.to_string(),
        &errore.to_string(),
        &memoria.to_string(),
    ]).unwrap();

    wtr.flush().unwrap();
}