
use std::process::Command;

pub fn get_nnz(path: &str) -> usize {
    let output = Command::new("python")
        .arg("get_nnz.py")
        .arg(path)
        .output()
        .expect("Cannot execute 'python' command.");

    if !output.status.success() {
        let error_msg = String::from_utf8_lossy(&output.stderr);
        panic!("Error:\n{}", error_msg);
    }

    let nnz_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    
    nnz_str.parse::<usize>().expect(&format!(
        "Cannot parse: '{}'", 
        nnz_str
    ))
}