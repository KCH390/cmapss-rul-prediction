/// Number of operational settings recorded per cycle (confirmed against readme.txt
/// and the raw files: columns 3-5).
pub const NUM_OP_SETTINGS: usize = 3;

/// Number of sensor channels recorded per cycle (columns 6-26). Note the readme
/// text labels these "sensor measurement 1" through "26" but there are 21 of
/// them once you account for the unit/cycle/op-setting columns — verified by
/// counting fields on real rows, not assumed from the readme's numbering.
pub const NUM_SENSORS: usize = 21;

/// Total whitespace-delimited fields expected per row.
pub const NUM_FIELDS: usize = 2 + NUM_OP_SETTINGS + NUM_SENSORS;

/// One engine-cycle snapshot: a single row of a train/test file.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleRecord {
    pub unit: u32,
    pub cycle: u32,
    pub op_settings: [f64; NUM_OP_SETTINGS],
    pub sensors: [f64; NUM_SENSORS],
}

/// Parses one raw line into a `CycleRecord`.
///
/// The files are whitespace-delimited with inconsistent spacing and a
/// trailing space on every line (confirmed by inspecting the raw bytes),
/// so this splits on any run of whitespace rather than assuming a fixed
/// delimiter or column width.
pub fn parse_line(line: &str) -> std::result::Result<CycleRecord, String> {
    let fields: Vec<&str> = line.split_whitespace().collect();

    if fields.len() != NUM_FIELDS {
        return Err(format!(
            "expected {} fields, found {}",
            NUM_FIELDS,
            fields.len()
        ));
    }

    let parse_f64 = |s: &str, label: &str| -> std::result::Result<f64, String> {
        s.parse::<f64>()
            .map_err(|e| format!("could not parse {} ({:?}) as f64: {}", label, s, e))
    };

    let unit = fields[0]
        .parse::<f64>()
        .map_err(|e| format!("could not parse unit ({:?}) : {}", fields[0], e))?
        as u32;
    let cycle = fields[1]
        .parse::<f64>()
        .map_err(|e| format!("could not parse cycle ({:?}): {}", fields[1], e))?
        as u32;

    let mut op_settings = [0.0f64; NUM_OP_SETTINGS];
    for (i, slot) in op_settings.iter_mut().enumerate() {
        *slot = parse_f64(fields[2 + i], "op_setting")?;
    }

    let mut sensors = [0.0f64; NUM_SENSORS];
    for (i, slot) in sensors.iter_mut().enumerate() {
        *slot = parse_f64(fields[2 + NUM_OP_SETTINGS + i], "sensor")?;
    }

    Ok(CycleRecord {
        unit,
        cycle,
        op_settings,
        sensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_real_looking_row() {
        let line = "1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 23.4190  ";
        let rec = parse_line(line).expect("should parse");
        assert_eq!(rec.unit, 1);
        assert_eq!(rec.cycle, 1);
        assert_eq!(rec.op_settings[0], -0.0007);
        assert_eq!(rec.sensors[0], 518.67);
        assert_eq!(rec.sensors[20], 23.4190);
    }

    #[test]
    fn rejects_wrong_field_count() {
        let line = "1 2 3";
        assert!(parse_line(line).is_err());
    }

    #[test]
    fn rejects_non_numeric_field() {
        let line = "1 1 -0.0007 -0.0004 100.0 518.67 641.82 1589.70 1400.60 14.62 21.61 554.36 2388.06 9046.19 1.30 47.47 521.66 2388.02 8138.62 8.4195 0.03 392 2388 100.00 39.06 NOTANUMBER";
        assert!(parse_line(line).is_err());
    }
}
