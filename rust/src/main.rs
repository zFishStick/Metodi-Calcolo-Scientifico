
use std::{fs::File, io::{BufRead, BufReader}, time::Instant};

use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::{CooMatrix, CscMatrix, factorization::CscCholesky};

fn main() {
    // Vettore soluzione di solo 1
    // let xe = DVector::from_element(matrix_a.ncols(), 1.0);

    // let time = Instant::now();
    // let x = cholesky_method(matrix_a, b);
    // print!("x: {:?}\n", x);

    // println!("Time: {} ms", time.elapsed().as_millis());

}

#[allow(dead_code)]
fn cholesky_method(matrix : CscMatrix<f64>, b : DVector<f64>) -> DVector<f64> {
    let factor = CscCholesky::factor(&matrix).unwrap();
    let result = factor.solve(&b);
    // Non c'è un return, e nemmeno il punto e virgola, la funzione restituisce direttamente il risultato dell'operazione
    DVector::from_vec(result.data.into())
}

#[allow(dead_code)]
fn get_sparse_matrix(path: &str) -> CscMatrix<f64> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|l| l.unwrap());

    // Salta l'header del file (commenti e cose varie)
    let header = loop {
        let line = lines.next().expect("File terminato prima dell'header");
        if !line.starts_with('%') {
            break line;
        }
    };

    let parts: Vec<usize> = header
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    // Il file .mtx ha una riga che specifica il numero di righe, colonne e non-zero entries (nnz)
    let (nrows, ncols, nnz) = (parts[0], parts[1], parts[2]);

    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut vals = Vec::with_capacity(nnz);

    // Da qui costruisce la matrice
    for line in lines {
        let line = line.trim().to_string();
        if line.starts_with('%') || line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let row = parts[0].parse::<usize>().unwrap() - 1;
        let col = parts[1].parse::<usize>().unwrap() - 1;
        let val = parts[2].parse::<f64>().unwrap();

        rows.push(row);
        cols.push(col);
        vals.push(val);

        if row != col {
            rows.push(col);
            cols.push(row);
            vals.push(val);
        }
    }

    // https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/
    // Coo è il tipo per le matrici sparse, usato per la costruzione della matrice
    let coo = CooMatrix::try_from_triplets(nrows, ncols, rows, cols, vals).unwrap();
    // Csc è un altro formato per le matrici sparse però più efficiente di Coo
    CscMatrix::from(&coo)
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

        let b = DVector::from_row_slice(&[44.0, 128.0, 214.0]);

        let x = cholesky_method(matrix_a, b);
        assert!(x.relative_eq(&DVector::from_row_slice(&[-8.0, 6.0, 2.0]), 1e-6, 1e-6));
    }

    #[test]
    fn test_cholesky_method_large_matrix() {
        // Il file arriva dal pdf del progetto
        let path = "C://Users//gabri//OneDrive//Desktop//ex15.mtx";
        let matrix : CscMatrix<f64> = get_sparse_matrix(path);

        let xe = DVector::from_element(matrix.ncols(), 1.0);
        let b : DVector<f64> = &matrix * &xe; // b = A * xe

        let x = cholesky_method(matrix, b);

        let diff = &x - &xe;
        let rel_error = diff.norm() / xe.norm();
        println!("Errore relativo: {}", rel_error);
        println!("x[0..5]: {:?}", &x.as_slice()[..5]);
        println!("xe[0..5]: {:?}", &xe.as_slice()[..5]);

        assert!(x.relative_eq(&xe, 1e-4, 1e-4)); //1e-4 sarebbe la vriabile tol
    }

}

