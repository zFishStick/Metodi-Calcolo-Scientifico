

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, factorization::CscCholesky};

mod matrix_extractor;
mod util;

fn main() {

}

#[allow(dead_code)]
fn cholesky_method(matrix : &CscMatrix<f64>, b : DVector<f64>) -> (CscCholesky<f64>, DVector<f64>) {
    let factor = CscCholesky::factor(&matrix).expect("La matrice non è simmetrica");
    let result = factor.solve(&b); // Ax = b
    (factor, DVector::from_vec(result.data.into()))
}

// Test
#[cfg(test)]
mod tests {
    
    use super::*;

    #[test]
    fn test_cholesky_method_small_matrix() {
        // https://numericalmethodsece101.weebly.com/choleskyrsquos-method.html
        let dense = DMatrix::from_row_slice(3, 3, 
            &[4.0, 10.0, 8.0,
            10.0, 26.0, 26.0,
            8.0, 26.0, 61.0]);
        
        let coo = CooMatrix::from(&dense);
        let matrix_a = CscMatrix::from(&coo);

        let matrix_t = CscCholesky::factor(&matrix_a).expect("La matrice non è simmetrica");
        let l = matrix_t.l();
        println!("L:\n{:?}", l);

        let b = DVector::from_row_slice(&[44.0, 128.0, 214.0]);

        println!("Matrix nnz: {}", matrix_a.nnz());
        println!("Matrix L nnz: {}", l.nnz());

        // Tipo f64 occupa 64 bit -> 8 byte
        let mem_occupata = (matrix_a.nnz() + l.nnz()) as f64 * 8.0;
        println!("Memoria occupata: {}", util::format_memory(mem_occupata as u64));

        let x = cholesky_method(&matrix_a, b);
        assert!(x.1.relative_eq(&DVector::from_row_slice(&[-8.0, 6.0, 2.0]), 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_method_large_matrix() {
        // Il file arriva dal pdf del progetto
        let path = "C://Users//gabri//OneDrive//Desktop//ex15.mtx";
        let matrix : CscMatrix<f64> = matrix_extractor::get_sparse_matrix(path);

        let xe = DVector::from_element(matrix.ncols(), 1.0);
        let b : DVector<f64> = &matrix * &xe; // b = A * xe

        let time = std::time::Instant::now();
        let (factor, x) = cholesky_method(&matrix, b);
        let elapsed = time.elapsed();

        let triangular_mat = factor.l();

        println!("Matrix nnz: {}", matrix.nnz());
        println!("Matrix L nnz: {}", triangular_mat.nnz());

        // Tipo f64 occupa 64 bit -> 8 byte
        let mem_occupata = (matrix.nnz() + triangular_mat.nnz()) as f64 * 8.0;
        println!("Memoria occupata: {}", util::format_memory(mem_occupata as u64));

        let diff = &x - &xe;
        let rel_error = diff.norm() / xe.norm();
        println!("Errore relativo: {}", rel_error);

        // Stampa i primi 5 elementi di x e xe per confronto
        println!("x[0..5]: {:?}", &x.as_slice()[..5]);
        println!("xe[0..5]: {:?}", &xe.as_slice()[..5]);
        println!("Tempo di esecuzione: {:.2?}", elapsed);

        assert!(x.relative_eq(&xe, 1e-4, 1e-4));
    }

}

