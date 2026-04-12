use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;

pub fn faer_get_sparse_matrix(path: &str) -> (usize, usize, Vec<(usize, usize, f64)>) {
    let file = File::open(path).expect("File non trovato");
    let reader = BufReader::new(file);
    let mut lines = reader.lines().map(|l| l.unwrap());

    // Salta l'header (commenti)
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
    
    let (nrows, ncols, nnz) = (parts[0], parts[1], parts[2]);

    let mut triplets = Vec::with_capacity(nnz * 2);

    for line in lines {
        let line = line.trim();
        if line.starts_with('%') || line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        let row = parts[0].parse::<usize>().unwrap() - 1;
        let col = parts[1].parse::<usize>().unwrap() - 1;
        let val = parts[2].parse::<f64>().unwrap();

        // Inseriamo l'elemento
        triplets.push((row, col, val));

        if row != col {
            triplets.push((col, row, val));
        }
    }

    (nrows, ncols, triplets)
}