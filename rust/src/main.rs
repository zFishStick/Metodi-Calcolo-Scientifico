mod util;

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
    //let folder = "C://Users//gabri//OneDrive//Desktop//matrici";

    let folder = "C:\\Users\\Simone\\Desktop\\Università\\Magistrale\\Metodi del calcolo scientifico\\Progetto1\\matrici_mtx";

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

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, SymbolicSparseColMat, Triplet};
    use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
    use faer::linalg::solvers::Solve;
    use faer::{Mat, Side};

    #[test]
    fn test_cholesky_method_small_matrix() {
        // 4.0  10.0   8.0
        // 10.0 26.0  26.0
        // 8.0  26.0  61.0
        let nrows = 3;
        let ncols = 3;

        let triplets = vec![
            Triplet { row: 0, col: 0, val: 4.0 },
            Triplet { row: 0, col: 1, val: 10.0 },
            Triplet { row: 0, col: 2, val: 8.0 },
            
            Triplet { row: 1, col: 0, val: 10.0 },
            Triplet { row: 1, col: 1, val: 26.0 },
            Triplet { row: 1, col: 2, val: 26.0 },
            
            Triplet { row: 2, col: 0, val: 8.0 },
            Triplet { row: 2, col: 1, val: 26.0 },
            Triplet { row: 2, col: 2, val: 61.0 },
        ];

        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &triplets)
            .expect("Errore");

        let mut b = Mat::<f64>::zeros(3, 1);
        b[(0, 0)] = 44.0;
        b[(1, 0)] = 128.0;
        b[(2, 0)] = 214.0;
        

        // Fattorizzazione Simbolica
        let symbolic = SymbolicLlt::<usize>::try_new(matrix.symbolic(), Side::Lower)
            .expect("Errore");
        
        // Fattorizzazione Numerica
        let factor = Llt::try_new_with_symbolic(symbolic, matrix.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");
        
        let x = factor.solve(b.as_ref());
        
        println!("Vettore x calcolato:\n{:?}", x);

        let mut x_esatto = Mat::<f64>::zeros(3, 1);
        x_esatto[(0, 0)] = -8.0;
        x_esatto[(1, 0)] = 6.0;
        x_esatto[(2, 0)] = 2.0;

        let diff = &x - &x_esatto;
        let errore = diff.norm_l2();
        println!("Errore assoluto rispetto alla soluzione: {:e}", errore);
        
        let size_of_val = std::mem::size_of::<f64>();
        let size_of_idx = std::mem::size_of::<usize>();

        let mem_occupata_a = (matrix.col_ptr().len() * size_of_idx)
            + (matrix.row_idx().len() * size_of_idx)
            + (matrix.val().len() * size_of_val);

        println!("Matrix A nnz: {}", matrix.val().len());
        println!("Memoria occupata matrice A: {} bytes", mem_occupata_a);
    }

    use sprs::io::read_matrix_market;

    #[test]
    fn test_cholesky_method_large_matrix() {
        let path = r"C:/Users/gabri/OneDrive/Desktop/cfd1.mtx";

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
        let b = &matrix_faer * &xe;

        let time = std::time::Instant::now();

        let symbolic = SymbolicLlt::<usize>::try_new(matrix_faer.symbolic(), Side::Lower)
            .expect("Errore simbolico");

        let factor = Llt::try_new_with_symbolic(symbolic, matrix_faer.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");

        let x = factor.solve(b.as_ref());

        let elapsed = time.elapsed();

        let size_of_val = std::mem::size_of::<f64>();
        let size_of_idx = std::mem::size_of::<usize>();
        let byte_for_val = (size_of_val + size_of_idx) as f64;

        let nnz_a = matrix_faer.compute_nnz();

        let nnz_l = util::get_nnz(path);

        let mem_occupata_mb = (nnz_a + nnz_l) as f64 * byte_for_val / (1024.0 * 1024.0);

        let diff = &x - &xe;
        let rel_error = diff.norm_l2() / xe.norm_l2();

        println!("NNZ A: {}, NNZ L: {}", nnz_a, nnz_l);
        println!("Memoria occupata (MB): {:.2} MB", mem_occupata_mb);
        println!("Errore relativo: {:e}", rel_error);
        println!("Tempo di esecuzione: {:.2?}", elapsed);

    }
}