//! Display utilities for the TTS server.
//!
//! Provides formatted console output for server status and progress.

use std::fmt;

/// ANSI color codes for terminal output
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const BLUE: &str = "\x1b[34m";
    pub const CYAN: &str = "\x1b[36m";
    pub const RED: &str = "\x1b[31m";
}

/// Status indicator for stage messages
#[derive(Debug, Clone, Copy)]
pub enum Status {
    Progress,
    Complete,
    Failed,
    Info,
}

impl Status {
    /// Get the colored symbol for this status
    pub fn symbol(&self) -> &'static str {
        match self {
            Status::Progress => concat!("\x1b[33m", "[...]", "\x1b[0m"),
            Status::Complete => concat!("\x1b[32m", "[OK]", "\x1b[0m"),
            Status::Failed => concat!("\x1b[31m", "[FAIL]", "\x1b[0m"),
            Status::Info => concat!("\x1b[34m", "[i]", "\x1b[0m"),
        }
    }
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}

/// Print a formatted stage message with status indicator
pub fn print_stage(message: &str, status: Status, elapsed: Option<f64>, extra: Option<&str>) {
    let mut parts = vec![status.symbol().to_string(), message.to_string()];
    
    if let Some(secs) = elapsed {
        parts.push(format!("{}({:.2}s){}", colors::DIM, secs, colors::RESET));
    }
    
    if let Some(msg) = extra {
        parts.push(format!("{}{}{}", colors::DIM, msg, colors::RESET));
    }
    
    println!("{}", parts.join(" "));
}

/// Format a duration in a human-readable way
pub fn format_duration(secs: f64) -> String {
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1000.0)
    } else if secs < 60.0 {
        format!("{:.2}s", secs)
    } else {
        let mins = (secs / 60.0).floor();
        let remaining = secs - mins * 60.0;
        format!("{:.0}m {:.1}s", mins, remaining)
    }
}

/// Format bytes in a human-readable way
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.5), "500ms");
        assert_eq!(format_duration(1.5), "1.50s");
        assert_eq!(format_duration(65.0), "1m 5.0s");
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1500), "1.46 KB");
        assert_eq!(format_bytes(1500000), "1.43 MB");
        assert_eq!(format_bytes(1500000000), "1.40 GB");
    }
}
