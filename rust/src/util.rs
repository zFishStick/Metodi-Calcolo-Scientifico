
pub fn format_memory(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"]; 
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < units.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, units[unit_idx])
}