mod util;

use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{SparseColMat, SymbolicSparseColMat};
use faer::linalg::solvers::Solve;
use faer::{Mat, Side};
use sprs::io::read_matrix_market;

fn main() {
    let folder = "C://Users//gabri//Desktop//matrici";

    let matrix_list = [
        "Flan_1565", "StocF-1465", "cfd2", "cfd1", "G3_circuit",
        "parabolic_fem", "apache2", "shallow_water1", "ex15",
    ];
    
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
        let b = &matrix_faer * &xe;
        
        let time = std::time::Instant::now();

        let symbolic = SymbolicLlt::<usize>::try_new(matrix_faer.symbolic(), Side::Lower)
            .expect("Errore");
        let factor = Llt::try_new_with_symbolic(symbolic, matrix_faer.as_ref(), Side::Lower)
            .expect("La matrice non è definita positiva o non è simmetrica");
        let x = factor.solve(b.as_ref());
        
        let elapsed = time.elapsed();

        let size_of_val = std::mem::size_of::<f64>();
        let size_of_idx = std::mem::size_of::<usize>();
        let byte_for_val = (size_of_val + size_of_idx) as f64;

        let nnz_a = matrix_sprs.nnz();

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
        let path = "C://Users//gabri//OneDrive//Desktop//ex15.mtx";

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

        let nnz_a = matrix_sprs.nnz();

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