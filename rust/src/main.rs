mod matrix_extractor;
mod util;

use cap::Cap;
use std::alloc;

#[global_allocator]
static ALLOCATOR: Cap<alloc::System> = Cap::new(alloc::System, usize::MAX);

use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, Triplet};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};

fn main() {
    let folder = "C://Users//gabri//Desktop//matrici";

    let matrix_list = [
        "Flan_1565", "StocF-1465", "cfd2", "cfd1", "G3_circuit",
        "parabolic_fem", "apache2", "shallow_water1", "ex15",
    ];
    
    for name in matrix_list {
        println!("\n--- Analisi Matrice: {} ---", name);
        let path = format!("{}/{}.mtx", folder, name);

        let (nrows, ncols, triplets) = matrix_extractor::faer_get_sparse_matrix(&path);

        let faer_triplets: Vec<Triplet<usize, usize, f64>> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet { row: r, col: c, val: v })
            .collect();

        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &faer_triplets)
            .expect("Errore nella costruzione della matrice CSC");

        let xe = Mat::<f64>::from_fn(ncols, 1, |_, _| 1.0);
        let b = &matrix * &xe;

        let mem_prima = ALLOCATOR.allocated();
        
        let time = std::time::Instant::now();

        let symbolic = SymbolicLlt::<usize>::try_new(matrix.symbolic(), Side::Lower)
            .expect("Errore");
        let factor = Llt::try_new_with_symbolic(symbolic, matrix.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");
        let x = factor.solve(b.as_ref());
        
        let elapsed = time.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();
        
        let mem_dopo = ALLOCATOR.allocated();
        
        // Memoria in MB
        let memoria_allocata_mb = mem_dopo.saturating_sub(mem_prima) as f64 / (1024.0 * 1024.0);

        let diff = &x - &xe;
        let rel_error = diff.norm_l2() / xe.norm_l2();

        println!("Tempo: {:.4} s", elapsed_secs);
        println!("Memoria allocata per risolvere: {:.2} MB", memoria_allocata_mb);
        println!("Errore relativo: {:e}", rel_error);

    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::util;
    use faer::sparse::{SparseColMat, Triplet};
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
        
        let mem_prima = ALLOCATOR.allocated();

        // Fattorizzazione Simbolica
        let symbolic = SymbolicLlt::<usize>::try_new(matrix.symbolic(), Side::Lower)
            .expect("Errore");
        
        // Fattorizzazione Numerica
        let factor = Llt::try_new_with_symbolic(symbolic, matrix.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");
        
        let x = factor.solve(b.as_ref());
        
        let mem_dopo = ALLOCATOR.allocated();
        let aumento_totale = mem_dopo.saturating_sub(mem_prima);

        println!("Vettore x calcolato:\n{:?}", x);

        let mut x_esatto = Mat::<f64>::zeros(3, 1);
        x_esatto[(0, 0)] = -8.0;
        x_esatto[(1, 0)] = 6.0;
        x_esatto[(2, 0)] = 2.0;

        let diff = &x - &x_esatto;
        let errore = diff.norm_l2();
        println!("Errore assoluto rispetto alla soluzione: {:e}", errore);
        
        assert!(errore < 1e-10, "Test fallito: errore troppo grande");
        let size_of_val = std::mem::size_of::<f64>();
        let size_of_idx = std::mem::size_of::<usize>();

        let mem_occupata_a = (matrix.col_ptr().len() * size_of_idx)
            + (matrix.row_idx().len() * size_of_idx)
            + (matrix.val().len() * size_of_val);

        println!("Matrix A nnz: {}", matrix.val().len());
        println!("Memoria occupata matrice A: {} bytes", mem_occupata_a);
        println!("Memoria totale allocata (L + buffer): {} bytes", aumento_totale);
    }

    #[test]
    fn test_cholesky_method_large_matrix() {
        // Il file arriva dal pdf del progetto
        let path = "C://Users//gabri//OneDrive//Desktop//ex15.mtx";
        
        let (nrows, ncols, triplets) = matrix_extractor::faer_get_sparse_matrix(path);

        let faer_triplets: Vec<Triplet<usize, usize, f64>> = triplets
            .iter()
            .map(|&(r, c, v)| Triplet { row: r, col: c, val: v })
            .collect();

        let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &faer_triplets)
            .expect("Errore");

        let xe = Mat::<f64>::from_fn(ncols, 1, |_, _| 1.0);
        
        let b = &matrix * &xe; 

        let mem_prima = ALLOCATOR.allocated();
        let time = std::time::Instant::now();

        let symbolic = SymbolicLlt::<usize>::try_new(matrix.symbolic(), Side::Lower)
            .expect("Errore");
        
        let factor = Llt::try_new_with_symbolic(symbolic, matrix.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");
        
        let x = factor.solve(b.as_ref());
        
        let elapsed = time.elapsed();
        let mem_dopo = ALLOCATOR.allocated();

        let aumento_totale = mem_dopo.saturating_sub(mem_prima);

        let size_of_val = std::mem::size_of::<f64>(); 
        let size_of_idx = std::mem::size_of::<usize>();

        // Non credo serva
        let mem_occupata_a = (matrix.col_ptr().len() * size_of_idx) + 
                             (matrix.row_idx().len() * size_of_idx) + 
                             (matrix.val().len() * size_of_val);

        let diff = &x - &xe;
        let rel_error = diff.norm_l2() / xe.norm_l2();

        println!("Matrix A nnz: {}", matrix.val().len());
        println!("Memoria occupata A: {}", util::format_memory(mem_occupata_a as u64));
        println!("Memoria totale allocata per risoluzione: {}", util::format_memory(aumento_totale as u64));
        println!("Errore relativo: {:e}", rel_error);
        println!("Tempo di esecuzione: {:.2?}", elapsed);

        assert!(rel_error < 1e-4, "Errore relativo alto");
    }
}