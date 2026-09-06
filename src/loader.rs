use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::error::{CmapssError, Result};
use crate::parser::{parse_line, CycleRecord};

/// One engine's full recorded trajectory, sorted by cycle.
#[derive(Debug, Clone)]
pub struct EngineRun {
    pub unit: u32,
    pub records: Vec<CycleRecord>,
}

impl EngineRun {
    /// Last recorded cycle number for this run.
    pub fn last_cycle(&self) -> u32 {
        // Safe: an EngineRun is never constructed with zero records (see group_by_unit).
        self.records.last().expect("EngineRun has no records").cycle
    }
}

/// Loads and parses every row of a train/test file into `CycleRecord`s,
/// in file order. Blank trailing lines are skipped; anything else that
/// fails to parse is a hard error with line number and raw content attached
/// rather than being silently dropped.
pub fn load_records(path: &Path) -> Result<Vec<CycleRecord>> {
    let content = fs::read_to_string(path).map_err(|source| CmapssError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut records = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = parse_line(line).map_err(|reason| CmapssError::Parse {
            path: path.to_path_buf(),
            line_number: idx + 1,
            raw_line: line.to_string(),
            reason,
        })?;
        records.push(record);
    }
    Ok(records)
}

/// Loads a `RUL_FDxxx.txt` ground-truth file: one integer RUL value per line,
/// in test-unit order (line 1 = unit 1's RUL at the end of its truncated
/// test trajectory, and so on).
pub fn load_rul(path: &Path) -> Result<Vec<u32>> {
    let content = fs::read_to_string(path).map_err(|source| CmapssError::Io {
        path: path.to_path_buf(),
        source,
    })?;

    let mut values = Vec::new();
    for (idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: u32 = trimmed.parse().map_err(|_| CmapssError::Parse {
            path: path.to_path_buf(),
            line_number: idx + 1,
            raw_line: line.to_string(),
            reason: "expected a single non-negative integer".to_string(),
        })?;
        values.push(value);
    }
    Ok(values)
}

/// Groups flat `CycleRecord`s by engine unit and sorts each group by cycle.
///
/// Uses a `BTreeMap` keyed on unit number so iteration order is deterministic
/// (unit 1, 2, 3, ...) regardless of the input file's row order — the raw
/// files are already unit-major and cycle-ordered, but this doesn't assume that.
pub fn group_by_unit(records: Vec<CycleRecord>) -> Vec<EngineRun> {
    let mut groups: BTreeMap<u32, Vec<CycleRecord>> = BTreeMap::new();
    for record in records {
        groups.entry(record.unit).or_default().push(record);
    }

    groups
        .into_iter()
        .map(|(unit, mut records)| {
            records.sort_by_key(|r| r.cycle);
            EngineRun { unit, records }
        })
        .collect()
}
