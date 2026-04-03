use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

use nalgebra_sparse::CscMatrix;
use nalgebra_sparse::CooMatrix;


#[allow(dead_code)]
pub fn get_sparse_matrix(path: &str) -> CscMatrix<f64> {
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