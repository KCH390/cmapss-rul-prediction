use std::fmt;
use std::path::PathBuf;

/// Crate-wide error type.
///
/// Kept as a small hand-rolled enum rather than pulling in `thiserror` — the
/// error surface here is narrow enough (I/O plus one parse failure mode)
/// that a dependency doesn't buy much.
#[derive(Debug)]
pub enum CmapssError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: PathBuf,
        line_number: usize,
        raw_line: String,
        reason: String,
    },
}

impl fmt::Display for CmapssError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmapssError::Io { path, source } => {
                write!(f, "I/O error reading {}: {}", path.display(), source)
            }
            CmapssError::Parse {
                path,
                line_number,
                raw_line,
                reason,
            } => write!(
                f,
                "parse error in {} at line {}: {} (raw line: {:?})",
                path.display(),
                line_number,
                reason,
                raw_line
            ),
        }
    }
}

impl std::error::Error for CmapssError {}

pub type Result<T> = std::result::Result<T, CmapssError>;
