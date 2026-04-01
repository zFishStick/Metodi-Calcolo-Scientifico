
pub fn format_memory(kb: u64) -> String {
    let units = ["KB", "MB", "GB", "TB"];
    let mut size = kb as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < units.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, units[unit_idx])
}