/// The four C-MAPSS sub-datasets. Metadata below is transcribed from the
/// official NASA PCoE `readme.txt` shipped with the dataset, not assumed —
/// see `data/raw/CMAPSSData/readme.txt` in this repo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subset {
    FD001,
    FD002,
    FD003,
    FD004,
}

impl Subset {
    pub fn all() -> [Subset; 4] {
        [Subset::FD001, Subset::FD002, Subset::FD003, Subset::FD004]
    }

    pub fn code(&self) -> &'static str {
        match self {
            Subset::FD001 => "FD001",
            Subset::FD002 => "FD002",
            Subset::FD003 => "FD003",
            Subset::FD004 => "FD004",
        }
    }

    /// Number of distinct operating conditions (1 = sea level only, 6 = full envelope).
    pub fn num_conditions(&self) -> u8 {
        match self {
            Subset::FD001 | Subset::FD003 => 1,
            Subset::FD002 | Subset::FD004 => 6,
        }
    }

    /// Number of fault modes present (1 = HPC degradation only, 2 = HPC + fan degradation).
    pub fn num_fault_modes(&self) -> u8 {
        match self {
            Subset::FD001 | Subset::FD002 => 1,
            Subset::FD003 | Subset::FD004 => 2,
        }
    }

    /// Expected number of engine units in the training split. Used by the
    /// integration tests as a sanity check against the real files — not used
    /// at runtime to gate anything.
    ///
    /// NOTE: the official readme's FD004 train/test counts (248 / 249) are
    /// transposed relative to the real files. Verified directly against the
    /// raw data: `train_FD004.txt` has 249 contiguous unit IDs (1..=249),
    /// `test_FD004.txt` has 248 (1..=248), and `RUL_FD004.txt` has exactly
    /// 248 lines, matching the test set. The totals agree (248 + 249 = 497
    /// either way), so this looks like the two numbers were swapped when the
    /// readme was written, not a data-completeness problem. These constants
    /// reflect the real files. Every other subset matches the readme exactly.
    pub fn expected_train_units(&self) -> usize {
        match self {
            Subset::FD001 => 100,
            Subset::FD002 => 260,
            Subset::FD003 => 100,
            Subset::FD004 => 249,
        }
    }

    /// Expected number of engine units in the test split. See the note on
    /// `expected_train_units` — FD004's readme-stated count is transposed
    /// with the train count relative to the actual files.
    pub fn expected_test_units(&self) -> usize {
        match self {
            Subset::FD001 => 100,
            Subset::FD002 => 259,
            Subset::FD003 => 100,
            Subset::FD004 => 248,
        }
    }

    pub fn train_path(&self, data_dir: &std::path::Path) -> std::path::PathBuf {
        data_dir.join(format!("train_{}.txt", self.code()))
    }

    pub fn test_path(&self, data_dir: &std::path::Path) -> std::path::PathBuf {
        data_dir.join(format!("test_{}.txt", self.code()))
    }

    pub fn rul_path(&self, data_dir: &std::path::Path) -> std::path::PathBuf {
        data_dir.join(format!("RUL_{}.txt", self.code()))
    }
}

impl std::fmt::Display for Subset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Parses a subset name from a CLI argument, case-insensitively, accepting
/// both "FD001" and "1" style shorthand.
impl std::str::FromStr for Subset {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "FD001" | "1" => Ok(Subset::FD001),
            "FD002" | "2" => Ok(Subset::FD002),
            "FD003" | "3" => Ok(Subset::FD003),
            "FD004" | "4" => Ok(Subset::FD004),
            other => Err(format!(
                "unrecognized subset {:?} (expected one of FD001, FD002, FD003, FD004)",
                other
            )),
        }
    }
}
