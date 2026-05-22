use serde::{Deserialize, Serialize};

use cold_sdk::{ChatMessage, ColdClient, Role, Usage};

use crate::boundary::{Boundaries, find_boundaries, snap_to_tool_groups};
use crate::config::CompressorConfig;
use crate::counter::{CharEstimator, TokenCounter};
use crate::error::ContextError;
use crate::integrity::{repair_sequence, validate_sequence};
use crate::microcompact;
use crate::note::build_compression_note;
use crate::pruner;
use crate::restore;
use crate::summary::{
    LlmSummarizer, SUMMARY_PREFIX, Summarizer, build_iterative_prompt, format_turns_for_summary,
};

/// Stages applied during compression, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressionStage {
    Microcompact,
    ImageStrip,
    ToolPrune,
    ToolDedup,
    ArgsTruncate,
    LlmSummarize,
    FileRestore,
}

/// Warnings that may accompany a successful compression.
#[derive(Debug, Clone)]
pub enum CompressionWarning {
    /// The LLM summary call failed; returning pruned messages without summary.
    SummaryFailed(String),
    /// Integrity issues were found and auto-repaired.
    IntegrityRepaired(usize),
    /// The security scan detected threats in the summary (number of threats found).
    InjectionDetected(usize),
}

/// Result of a compression operation.
#[derive(Debug)]
pub struct CompressionResult {
    /// The compressed message list.
    pub messages: Vec<ChatMessage>,
    /// Stages that were applied.
    pub stages: Vec<CompressionStage>,
    /// Any warnings.
    pub warnings: Vec<CompressionWarning>,
    /// Estimated token savings percentage (0.0..1.0).
    pub savings_pct: f32,
    /// Original estimated token count.
    pub original_tokens: u32,
    /// Final estimated token count.
    pub final_tokens: u32,
    /// A note to include in the system prompt after compression, if configured.
    pub note: Option<String>,
}

/// Serializable snapshot of the compressor's runtime state.
///
/// Enables the caller to persist state across process restarts via
/// [`ContextCompressor::save_state`] and [`ContextCompressor::restore_state`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressorState {
    /// Last known prompt token count.
    pub last_prompt_tokens: u32,
    /// Last known completion token count.
    pub last_completion_tokens: u32,
    /// How many times LLM summarization has been performed.
    pub compression_count: u32,
    /// The raw summary text from the most recent LLM summarization.
    pub previous_summary: Option<String>,
    /// Consecutive ineffective compression count (anti-thrashing).
    pub ineffective_count: u32,
    /// Savings percentage from the most recent compression.
    pub last_savings_pct: f32,
}

/// The main context compressor. Progressively applies pruning stages and, if needed,
/// an LLM-powered summarization to keep the conversation within the context window.
pub struct ContextCompressor<S: Summarizer = LlmSummarizer> {
    config: CompressorConfig,
    counter: CharEstimator,
    summarizer: S,

    // Runtime state
    last_prompt_tokens: u32,
    last_completion_tokens: u32,
    threshold_tokens: u32,
    compression_count: u32,
    previous_summary: Option<String>,
    ineffective_count: u32,
    last_savings_pct: f32,
}

impl ContextCompressor<LlmSummarizer> {
    /// Create a new compressor with the default `LlmSummarizer`.
    #[must_use]
    pub fn new(config: CompressorConfig, client: ColdClient) -> Self {
        let model = config.model.clone();
        let threshold_tokens = config.threshold_tokens();
        Self {
            config,
            counter: CharEstimator,
            summarizer: LlmSummarizer::new(client, model),
            last_prompt_tokens: 0,
            last_completion_tokens: 0,
            threshold_tokens,
            compression_count: 0,
            previous_summary: None,
            ineffective_count: 0,
            last_savings_pct: 0.0,
        }
    }
}

/// Minimum absolute token growth required to reset anti-thrashing.
const ANTI_THRASH_MIN_GROWTH: u32 = 1000;

/// Minimum post-pruning middle-zone token count below which LLM summarization is skipped.
const MIN_MIDDLE_TOKENS_FOR_SUMMARY: u32 = 2000;

impl<S: Summarizer> ContextCompressor<S> {
    /// Create a compressor with a custom summarizer (useful for testing).
    #[must_use]
    pub fn with_summarizer(config: CompressorConfig, summarizer: S) -> Self {
        let threshold_tokens = config.threshold_tokens();
        Self {
            config,
            counter: CharEstimator,
            summarizer,
            last_prompt_tokens: 0,
            last_completion_tokens: 0,
            threshold_tokens,
            compression_count: 0,
            previous_summary: None,
            ineffective_count: 0,
            last_savings_pct: 0.0,
        }
    }

    /// How many times the compressor has successfully performed LLM summarization.
    #[must_use]
    pub const fn compression_count(&self) -> u32 {
        self.compression_count
    }

    /// The last known prompt token count from [`update_usage`](Self::update_usage).
    #[must_use]
    pub const fn last_prompt_tokens(&self) -> u32 {
        self.last_prompt_tokens
    }

    /// The last known completion token count from [`update_usage`](Self::update_usage).
    #[must_use]
    pub const fn last_completion_tokens(&self) -> u32 {
        self.last_completion_tokens
    }

    /// The savings percentage from the most recent compression (0.0..1.0).
    #[must_use]
    pub const fn last_savings_pct(&self) -> f32 {
        self.last_savings_pct
    }

    /// The raw summary text from the most recent LLM summarization, if any.
    #[must_use]
    pub fn previous_summary(&self) -> Option<&str> {
        self.previous_summary.as_deref()
    }

    /// Reset the compressor state so it can be reused for a new conversation.
    pub fn reset(&mut self) {
        self.last_prompt_tokens = 0;
        self.last_completion_tokens = 0;
        self.previous_summary = None;
        self.compression_count = 0;
        self.ineffective_count = 0;
        self.last_savings_pct = 0.0;
    }

    /// Capture the compressor's runtime state as a serializable snapshot.
    #[must_use]
    pub fn save_state(&self) -> CompressorState {
        CompressorState {
            last_prompt_tokens: self.last_prompt_tokens,
            last_completion_tokens: self.last_completion_tokens,
            compression_count: self.compression_count,
            previous_summary: self.previous_summary.clone(),
            ineffective_count: self.ineffective_count,
            last_savings_pct: self.last_savings_pct,
        }
    }

    /// Restore the compressor's runtime state from a previously saved snapshot.
    pub fn restore_state(&mut self, state: CompressorState) {
        self.last_prompt_tokens = state.last_prompt_tokens;
        self.last_completion_tokens = state.last_completion_tokens;
        self.compression_count = state.compression_count;
        self.previous_summary = state.previous_summary;
        self.ineffective_count = state.ineffective_count;
        self.last_savings_pct = state.last_savings_pct;
    }

    /// Update tracked token usage from an API response.
    ///
    /// The anti-thrashing counter is reset only when `prompt_tokens` exceeds the
    /// previous value by more than 20% **and** the absolute growth is at least
    /// `ANTI_THRASH_MIN_GROWTH` (1000 tokens).
    pub fn update_usage(&mut self, usage: &Usage) {
        if self.last_prompt_tokens > 0 {
            let growth = f64::from(usage.prompt_tokens) / f64::from(self.last_prompt_tokens);
            let abs_growth = usage.prompt_tokens.saturating_sub(self.last_prompt_tokens);
            if growth > 1.20 && abs_growth >= ANTI_THRASH_MIN_GROWTH {
                self.ineffective_count = 0;
            }
        }
        self.last_prompt_tokens = usage.prompt_tokens;
        self.last_completion_tokens = usage.completion_tokens;
    }

    /// Manually reset the anti-thrashing counter.
    pub const fn reset_anti_thrashing(&mut self) {
        self.ineffective_count = 0;
    }

    /// Check whether compression should be triggered based on token usage.
    #[must_use]
    pub const fn should_compress(&self) -> bool {
        if self.ineffective_count >= 2 {
            return false;
        }
        self.last_prompt_tokens >= self.threshold_tokens
    }

    /// Quick char-based preflight check (no API usage data needed).
    #[must_use]
    pub fn should_compress_preflight(&self, messages: &[ChatMessage]) -> bool {
        if self.ineffective_count >= 2 {
            return false;
        }
        let estimated = self.counter.count_messages(messages);
        estimated >= self.threshold_tokens
    }

    /// Run the progressive compression pipeline.
    ///
    /// # Errors
    ///
    /// Returns `ContextError::NothingToCompress` if there are no middle-zone messages.
    /// Returns `ContextError::Summarization` only if all stages fail.
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub async fn compress(
        &mut self,
        messages: Vec<ChatMessage>,
        focus_topic: Option<&str>,
    ) -> Result<CompressionResult, ContextError> {
        let original_tokens = self.counter.count_messages(&messages);

        // Find boundaries
        let boundaries = find_boundaries(&messages, &self.config, &self.counter);
        let boundaries = snap_to_tool_groups(&messages, boundaries);
        let Boundaries {
            head_end,
            tail_start,
        } = boundaries;

        if head_end >= tail_start {
            return Err(ContextError::NothingToCompress);
        }

        let original_messages = messages.clone();
        let mut current = messages;
        let mut stages: Vec<CompressionStage> = Vec::new();
        let mut warnings: Vec<CompressionWarning> = Vec::new();
        let mut running_tokens = original_tokens;

        // Stage -1: Microcompact (clear stale tool results)
        if let Some(ref mc_config) = self.config.microcompact {
            let (new_msgs, count, delta) = microcompact::microcompact(current, mc_config);
            current = new_msgs;
            if count > 0 {
                running_tokens = running_tokens.saturating_sub(delta);
                stages.push(CompressionStage::Microcompact);
                if running_tokens < self.threshold_tokens {
                    return Ok(self.build_result(current, stages, warnings, original_tokens));
                }
            }
        }

        // Stage 0: Strip historical images
        {
            let (new_msgs, count, delta) = pruner::strip_historical_images(current, tail_start);
            current = new_msgs;
            if count > 0 {
                running_tokens = running_tokens.saturating_sub(delta);
                stages.push(CompressionStage::ImageStrip);
                if running_tokens < self.threshold_tokens {
                    return Ok(self.build_result(current, stages, warnings, original_tokens));
                }
            }
        }

        // Stage 1: Prune tool outputs
        {
            let (new_msgs, count, delta) =
                pruner::prune_tool_outputs(current, head_end, tail_start);
            current = new_msgs;
            if count > 0 {
                running_tokens = running_tokens.saturating_sub(delta);
                stages.push(CompressionStage::ToolPrune);
                if running_tokens < self.threshold_tokens {
                    return Ok(self.build_result(current, stages, warnings, original_tokens));
                }
            }
        }

        // Stage 2: Dedup tool results
        {
            let (new_msgs, count, delta) =
                pruner::dedup_tool_results(current, head_end, tail_start);
            current = new_msgs;
            if count > 0 {
                running_tokens = running_tokens.saturating_sub(delta);
                stages.push(CompressionStage::ToolDedup);
                if running_tokens < self.threshold_tokens {
                    return Ok(self.build_result(current, stages, warnings, original_tokens));
                }
            }
        }

        // Stage 3: Truncate tool args
        {
            let (new_msgs, count, delta) =
                pruner::truncate_tool_args(current, head_end, tail_start);
            current = new_msgs;
            if count > 0 {
                running_tokens = running_tokens.saturating_sub(delta);
                stages.push(CompressionStage::ArgsTruncate);
                if running_tokens < self.threshold_tokens {
                    return Ok(self.build_result(current, stages, warnings, original_tokens));
                }
            }
        }

        // Stage 4: LLM summarize
        {
            let middle_tokens = self.counter.count_messages(&current[head_end..tail_start]);
            if middle_tokens < MIN_MIDDLE_TOKENS_FOR_SUMMARY {
                return Ok(self.build_result(current, stages, warnings, original_tokens));
            }

            let middle = &current[head_end..tail_start];
            let mut turns_text = format_turns_for_summary(middle);

            if self.config.redact_sensitive {
                turns_text = crate::redact::redact_sensitive(&turns_text);
            }

            let mut prompt_content = if let Some(ref prev) = self.previous_summary {
                build_iterative_prompt(prev, &turns_text)
            } else {
                turns_text
            };

            if let Some(topic) = focus_topic {
                use std::fmt::Write as _;
                let _ = write!(
                    prompt_content,
                    "\n\nFocus especially on information related to: {topic}"
                );
            }

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let max_summary_tokens =
                (f64::from(self.threshold_tokens) * f64::from(self.config.summary_ratio)) as u32;
            let max_summary_tokens = max_summary_tokens.max(256);

            match self
                .summarizer
                .summarize(&prompt_content, max_summary_tokens)
                .await
            {
                Ok(mut summary_text) => {
                    stages.push(CompressionStage::LlmSummarize);

                    if self.config.redact_sensitive {
                        summary_text = crate::redact::redact_sensitive(&summary_text);
                    }

                    if self.config.scan_injections {
                        let scan = crate::security::scan_content(&summary_text);
                        if !scan.is_safe() {
                            warnings
                                .push(CompressionWarning::InjectionDetected(scan.threats.len()));
                        }
                    }

                    // Remove old summary + ack messages from head
                    let mut result_msgs =
                        Vec::with_capacity(head_end + 2 + (current.len() - tail_start));

                    for msg in &current[..head_end] {
                        if is_summary_message(msg) || is_ack_message(msg) {
                            continue;
                        }
                        result_msgs.push(msg.clone());
                    }

                    // Strip prefix if model echoed it back
                    let clean = summary_text
                        .strip_prefix(SUMMARY_PREFIX)
                        .map_or(summary_text.as_str(), str::trim_start);
                    let full_summary = format!("{SUMMARY_PREFIX}\n\n{clean}");
                    result_msgs.push(ChatMessage::user(&full_summary));

                    // Stage 5: File restoration — re-inject recently-read files
                    if let Some(ref restore_config) = self.config.restore {
                        let recent_paths = restore::find_recent_reads(
                            &original_messages,
                            restore_config.max_files,
                        );
                        if !recent_paths.is_empty() {
                            let restoration_msgs = restore::build_restoration_messages(
                                &recent_paths,
                                restore_config,
                                &self.config.project_root,
                            )
                            .await;
                            if !restoration_msgs.is_empty() {
                                // Insert ack before restoration if needed to avoid
                                // consecutive user messages
                                if result_msgs
                                    .last()
                                    .is_some_and(|m| m.role == Role::User)
                                {
                                    result_msgs.push(ChatMessage::assistant(ACK_TEXT));
                                }
                                result_msgs.extend(restoration_msgs);
                                stages.push(CompressionStage::FileRestore);
                            }
                        }
                    }

                    // Ack before tail if needed
                    if result_msgs
                        .last()
                        .is_some_and(|m| m.role == Role::User)
                        && current
                            .get(tail_start)
                            .is_some_and(|m| m.role == Role::User)
                    {
                        result_msgs.push(ChatMessage::assistant(ACK_TEXT));
                    }

                    result_msgs.extend_from_slice(&current[tail_start..]);

                    // Validate and repair
                    let issues = validate_sequence(&result_msgs);
                    if !issues.is_empty() {
                        let issue_count = issues.len();
                        result_msgs = repair_sequence(result_msgs);
                        warnings.push(CompressionWarning::IntegrityRepaired(issue_count));
                    }

                    self.previous_summary = Some(summary_text);
                    self.compression_count += 1;

                    let final_tokens = self.counter.count_messages(&result_msgs);
                    let savings = if original_tokens > 0 {
                        1.0 - (f64::from(final_tokens) / f64::from(original_tokens))
                    } else {
                        0.0
                    };

                    #[allow(clippy::cast_possible_truncation)]
                    let savings_f32 = savings as f32;

                    if savings_f32 < 0.10 {
                        self.ineffective_count += 1;
                    } else {
                        self.ineffective_count = 0;
                    }
                    self.last_savings_pct = savings_f32;

                    let note = if self.config.compression_note {
                        Some(build_compression_note(self.compression_count))
                    } else {
                        None
                    };

                    return Ok(CompressionResult {
                        messages: result_msgs,
                        stages,
                        warnings,
                        savings_pct: savings_f32,
                        original_tokens,
                        final_tokens,
                        note,
                    });
                }
                Err(e) => {
                    warnings.push(CompressionWarning::SummaryFailed(e.to_string()));
                }
            }
        }

        Ok(self.build_result(current, stages, warnings, original_tokens))
    }

    fn build_result(
        &mut self,
        messages: Vec<ChatMessage>,
        stages: Vec<CompressionStage>,
        warnings: Vec<CompressionWarning>,
        original_tokens: u32,
    ) -> CompressionResult {
        let final_tokens = self.counter.count_messages(&messages);
        let savings = if original_tokens > 0 {
            1.0 - (f64::from(final_tokens) / f64::from(original_tokens))
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let savings_f32 = savings as f32;
        self.last_savings_pct = savings_f32;

        // Update anti-thrashing on ALL return paths, not just LLM summarize
        if savings_f32 < 0.10 {
            self.ineffective_count += 1;
        } else {
            self.ineffective_count = 0;
        }

        CompressionResult {
            messages,
            stages,
            warnings,
            savings_pct: savings_f32,
            original_tokens,
            final_tokens,
            note: None,
        }
    }
}

const ACK_TEXT: &str = "Understood. Resuming from the latest context.";

fn is_summary_message(msg: &ChatMessage) -> bool {
    use cold_sdk::MessageContent;
    match msg.content {
        Some(MessageContent::Text(ref t)) => t.starts_with(SUMMARY_PREFIX),
        Some(MessageContent::Parts(ref parts)) => parts.first().is_some_and(|p| {
            if let cold_sdk::ContentPart::Text { text } = p {
                text.starts_with(SUMMARY_PREFIX)
            } else {
                false
            }
        }),
        None => false,
    }
}

fn is_ack_message(msg: &ChatMessage) -> bool {
    use cold_sdk::MessageContent;
    msg.role == Role::Assistant
        && matches!(&msg.content, Some(MessageContent::Text(t)) if t == ACK_TEXT)
}
