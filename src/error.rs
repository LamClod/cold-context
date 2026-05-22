use std::fmt;

use crate::security::Threat;

/// Errors that can occur during context compression.
#[derive(Debug)]
pub enum ContextError {
    /// The LLM summarization call failed.
    Summarization(cold_sdk::ColdError),
    /// The LLM returned an empty or invalid summary.
    EmptySummary(String),
    /// There are no messages in the middle zone to compress.
    NothingToCompress,
    /// The compression was aborted (e.g. by cancellation).
    Aborted,
    /// The security scanner detected threats in content.
    InjectionDetected(Vec<Threat>),
    /// Reactive compact could not reduce tokens below the target.
    ReactiveCompactFailed(String),
}

impl fmt::Display for ContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Summarization(e) => write!(f, "summarization failed: {e}"),
            Self::EmptySummary(msg) => write!(f, "empty summary: {msg}"),
            Self::NothingToCompress => write!(f, "nothing to compress"),
            Self::Aborted => write!(f, "compression aborted"),
            Self::InjectionDetected(threats) => {
                write!(f, "injection detected: {} threat(s)", threats.len())
            }
            Self::ReactiveCompactFailed(msg) => {
                write!(f, "reactive compact failed: {msg}")
            }
        }
    }
}

impl std::error::Error for ContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Summarization(e) => Some(e),
            _ => None,
        }
    }
}

impl From<cold_sdk::ColdError> for ContextError {
    fn from(e: cold_sdk::ColdError) -> Self {
        Self::Summarization(e)
    }
}
