

use nalgebra::{DVector};
use nalgebra_sparse::{CscMatrix, factorization::CscCholesky};
use std::fs::File;
use std::fs;
use csv::Writer;


mod matrix_extractor;
mod util;

fn main() {
    let paths: Vec<_> = fs::read_dir("C:\\Users\\Diagon\\Desktop\\UNIMIB\\ANNO 1\\SECONDO SEMESTRE\\Metodi Calcolo\\Matrici-Sparse")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.unwrap().path();
            if path.extension()? == "mtx" {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    for path in paths {

        let matrix: CscMatrix<f64> = matrix_extractor::get_sparse_matrix(&path.to_string_lossy());

        let xe = DVector::from_element(matrix.ncols(), 1.0);
        let b = &matrix * &xe;

        let time = std::time::Instant::now();
        let (factor, x) = cholesky_method(&matrix, b);
        let elapsed = time.elapsed();

        let triangular_mat = factor.l();

        let size_of_val = std::mem::size_of::<f64>(); 
        let size_of_idx = std::mem::size_of::<usize>();

        let mem_occupata_l = ((triangular_mat.ncols() + 1) * size_of_idx) + 
                             (triangular_mat.nnz() * size_of_idx) + 
                             (triangular_mat.nnz() * size_of_val);

        let memoria_occupata_x = x.len() * size_of_val;

        let memoria_mb = (mem_occupata_l + memoria_occupata_x) as f64 / (1024.0 * 1024.0);

        let diff = &x - &xe;
        let rel_error = diff.norm() / xe.norm();

        let nome = std::path::Path::new(&path)
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap();

        write_results_csv(
            "risultati_rust.csv",
            nome,
            matrix.ncols(),
            elapsed.as_secs_f64(),
            rel_error,
            memoria_mb
        );
    }
}

#[allow(dead_code)]
fn cholesky_method(matrix : &CscMatrix<f64>, b : DVector<f64>) -> (CscCholesky<f64>, DVector<f64>) {
    let factor = CscCholesky::factor(&matrix).expect("La matrice non è simmetrica");
    let result = factor.solve(&b); // Ax = b
    (factor, DVector::from_vec(result.data.into()))
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

// Test
#[cfg(test)]
mod tests {
    
    use super::*;
    use nalgebra::{DMatrix};
    use nalgebra_sparse::{CooMatrix};

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

        let size_of_val = std::mem::size_of::<f64>(); 
        let size_of_idx = std::mem::size_of::<usize>();

        // Matrice A
        let mem_colonne = (matrix_a.ncols() + 1) * size_of_idx;
        let mem_righe = matrix_a.nnz() * size_of_idx;         
        let mem_valori = matrix_a.nnz() * size_of_val;
        let mem_occupata_a = mem_colonne + mem_righe + mem_valori;

        // Matrice L
        let mem_colonne_l = (l.ncols() + 1) * size_of_idx;
        let mem_righe_l = l.nnz() * size_of_idx;
        let mem_valori_l = l.nnz() * size_of_val;
        let mem_occupata_l = mem_colonne_l + mem_righe_l + mem_valori_l;

        let x = cholesky_method(&matrix_a, b);

        let memoria_occupata_x = x.1.len() * size_of_val;
        println!("Memoria occupata vettore x: {}", util::format_memory(memoria_occupata_x as u64));

        println!("Memoria occupata matrice A: {}", util::format_memory((mem_colonne + mem_righe + mem_valori) as u64));
        println!("Memoria occupata matrice L: {}", util::format_memory((mem_colonne_l + mem_righe_l + mem_valori_l) as u64));

        let aumento_totale = mem_occupata_l + memoria_occupata_x;
        println!("Memoria totale occupata: {}", util::format_memory(aumento_totale as u64));

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
        let size_of_val = std::mem::size_of::<f64>(); 
        let size_of_idx = std::mem::size_of::<usize>();

        // https://docs.rs/nalgebra-sparse/latest/nalgebra_sparse/csc/struct.CscMatrix.html

        // Matrice A (Matrice originale)
        let mem_occupata_a = ((matrix.ncols() + 1) * size_of_idx) +  // Col offset
                            (matrix.nnz() * size_of_idx) + // Row indices
                            (matrix.nnz() * size_of_val); // Values

        // Matrice L (Matrice triangolare inferiore)
        let mem_occupata_l = ((triangular_mat.ncols() + 1) * size_of_idx) + 
                            (triangular_mat.nnz() * size_of_idx) + 
                            (triangular_mat.nnz() * size_of_val);

        let memoria_occupata_x = x.len() * size_of_val;

        println!("Matrix nnz: {}", matrix.nnz());
        println!("Matrix L nnz: {}", triangular_mat.nnz());
        println!("Memoria A: {}", util::format_memory(mem_occupata_a as u64));
        println!("Memoria L: {}", util::format_memory(mem_occupata_l as u64));
        println!("Memoria x: {}", util::format_memory(memoria_occupata_x as u64));
        
        let aumento_totale = mem_occupata_l + memoria_occupata_x;
        println!("Memoria totale occupata: {}", util::format_memory(aumento_totale as u64));

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

