use nalgebra_sparse::{self, CscMatrix, factorization::CscCholesky};
use sprs::io::read_matrix_market;

#[allow(dead_code)]
pub fn get_nnz(path: &str) -> usize {
    let matrix_sprs = read_matrix_market::<f64, usize, _>(path)
        .expect("Errore nella lettura del file Matrix Market")
        .to_csc();

    let nrows = matrix_sprs.rows();
    let ncols = matrix_sprs.cols();

    let (indptr, indices, data) = matrix_sprs.into_raw_storage();

    let matrix_nalgebra = CscMatrix::try_from_csc_data(
        nrows,
        ncols,
        indptr.to_vec(),
        indices.to_vec(),
        data.to_vec(),
    ).expect("Errore");

    let cholesky = CscCholesky::factor(&matrix_nalgebra)
        .expect("Errore");

    cholesky.l().nnz()
}