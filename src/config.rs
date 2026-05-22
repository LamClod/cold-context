//! Configuration for the context compressor.

use std::path::PathBuf;

use crate::budget::BudgetConfig;
use crate::microcompact::MicrocompactConfig;
use crate::restore::RestoreConfig;

#[derive(Debug, Clone)]
pub struct CompressorConfig {
    /// Model identifier used for summarization.
    pub model: String,
    /// Total context length (in tokens) of the target model.
    pub context_length: u32,
    /// Fraction of conversation budget that triggers compression (default 0.50).
    pub threshold_percent: f32,
    /// Number of non-system messages at the start to protect (default 3).
    pub protect_first_n: usize,
    /// Number of messages at the end to protect (default 6).
    pub protect_last_n: usize,
    /// Fraction of threshold tokens allocated for the summary output (default 0.20).
    pub summary_ratio: f32,
    /// Token budget allocation. If `None`, uses `BudgetConfig::default()`.
    pub budget: Option<BudgetConfig>,
    /// Whether to redact sensitive information before summarization (default true).
    pub redact_sensitive: bool,
    /// Whether to scan summaries for prompt injection (default true).
    pub scan_injections: bool,
    /// Whether to include a compression note in the result (default true).
    pub compression_note: bool,
    /// Microcompact config. If `Some`, old tool results are cleared before compression.
    pub microcompact: Option<MicrocompactConfig>,
    /// File restoration config. If `Some`, recently-read files are re-injected after
    /// LLM summarization.
    pub restore: Option<RestoreConfig>,
    /// Project root directory for file restoration. Defaults to current directory.
    pub project_root: PathBuf,
}

/// Clamp a value to the given inclusive range. NaN defaults to `min`.
fn clamp_f32(value: f32, min: f32, max: f32) -> f32 {
    if value.is_nan() || value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

impl CompressorConfig {
    /// Create a new config.
    ///
    /// `context_length` is the model's total context window in tokens
    /// (e.g., `128_000` for GPT-4o, `200_000` for Claude Sonnet 4).
    /// This value should be provided by the agent UI / caller.
    ///
    /// # Panics
    ///
    /// Panics if `context_length` is 0.
    #[must_use]
    pub fn new(model: impl Into<String>, context_length: u32) -> Self {
        assert!(context_length > 0, "context_length must be > 0");
        Self {
            model: model.into(),
            context_length,
            threshold_percent: 0.50,
            protect_first_n: 3,
            protect_last_n: 6,
            summary_ratio: 0.20,
            budget: None,
            redact_sensitive: true,
            scan_injections: true,
            compression_note: true,
            microcompact: Some(MicrocompactConfig::default()),
            restore: Some(RestoreConfig::default()),
            project_root: PathBuf::from("."),
        }
    }

    /// Clamped to `0.05..=1.0`.
    #[must_use]
    pub fn with_threshold_percent(mut self, pct: f32) -> Self {
        self.threshold_percent = clamp_f32(pct, 0.05, 1.0);
        self
    }

    #[must_use]
    pub const fn with_protect_first_n(mut self, n: usize) -> Self {
        self.protect_first_n = n;
        self
    }

    #[must_use]
    pub const fn with_protect_last_n(mut self, n: usize) -> Self {
        self.protect_last_n = n;
        self
    }

    /// Clamped to `0.05..=0.80`.
    #[must_use]
    pub fn with_summary_ratio(mut self, ratio: f32) -> Self {
        self.summary_ratio = clamp_f32(ratio, 0.05, 0.80);
        self
    }

    #[must_use]
    pub const fn with_budget(mut self, budget: BudgetConfig) -> Self {
        self.budget = Some(budget);
        self
    }

    #[must_use]
    pub const fn with_redact_sensitive(mut self, redact: bool) -> Self {
        self.redact_sensitive = redact;
        self
    }

    #[must_use]
    pub const fn with_scan_injections(mut self, scan: bool) -> Self {
        self.scan_injections = scan;
        self
    }

    #[must_use]
    pub const fn with_compression_note(mut self, note: bool) -> Self {
        self.compression_note = note;
        self
    }

    #[must_use]
    pub const fn with_microcompact(mut self, config: Option<MicrocompactConfig>) -> Self {
        self.microcompact = config;
        self
    }

    #[must_use]
    pub const fn with_restore(mut self, config: Option<RestoreConfig>) -> Self {
        self.restore = config;
        self
    }

    #[must_use]
    pub fn with_project_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.project_root = root.into();
        self
    }

    /// Compute the threshold token count.
    ///
    /// When a [`BudgetConfig`] is set, the threshold is based on the
    /// **conversation** budget (total minus system/tools/completion reserves),
    /// not the raw `context_length`. This ensures compression triggers
    /// relative to the space actually available for conversation history.
    #[must_use]
    pub fn threshold_tokens(&self) -> u32 {
        let base = self.budget.as_ref().map_or(self.context_length, |budget| {
            crate::budget::TokenBudget::new(self.context_length, budget).conversation
        });
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tokens = (f64::from(base) * f64::from(self.threshold_percent)) as u32;
        tokens
    }
}
