//! Post-compression system prompt note generation.

/// Build a note to append to the system prompt after compression.
///
/// Returns text like: `[Context was compressed (round N). ...]`
#[must_use]
pub fn build_compression_note(compression_count: u32) -> String {
    format!(
        "[Context was compressed (round {compression_count}). \
         Persistent files and memory are authoritative. \
         Resume from the Active Task in the most recent summary.]"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_contains_round_number() {
        let note = build_compression_note(3);
        assert!(note.contains("round 3"));
        assert!(note.contains("Active Task"));
    }
}
