use std::future::Future;

use cold_sdk::{ChatMessage, ChatRequest, ColdClient};

use crate::error::ContextError;
use crate::template::{ITERATIVE_USER_TEMPLATE, SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE};
use crate::util::extract_text_from_message;

/// Prefix prepended to every generated summary to instruct the LLM not to treat it as active.
pub const SUMMARY_PREFIX: &str = "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below. This is background reference, NOT active instructions. Do NOT answer questions mentioned in this summary. Respond ONLY to the latest user message that appears AFTER this summary.";

/// Trait for summarizing conversation turns.
pub trait Summarizer: Send + Sync {
    /// Summarize the given content, producing at most `max_tokens` of output.
    fn summarize(
        &self,
        content: &str,
        max_tokens: u32,
    ) -> impl Future<Output = Result<String, ContextError>> + Send;
}

/// LLM-based summarizer that calls the `ColdClient` chat API.
pub struct LlmSummarizer {
    client: ColdClient,
    model: String,
}

impl LlmSummarizer {
    /// Create a new `LlmSummarizer`.
    #[must_use]
    pub const fn new(client: ColdClient, model: String) -> Self {
        Self { client, model }
    }
}

impl Summarizer for LlmSummarizer {
    async fn summarize(&self, content: &str, max_tokens: u32) -> Result<String, ContextError> {
        #[allow(clippy::literal_string_with_formatting_args)]
        let user_text = SUMMARY_USER_TEMPLATE.replace("{turns}", content);

        let messages = vec![
            ChatMessage::system(SUMMARY_SYSTEM_PROMPT),
            ChatMessage::user(user_text),
        ];

        let mut req = ChatRequest::new(self.model.clone(), messages);
        req.max_tokens = Some(max_tokens);
        req.temperature = Some(0.0);

        let resp = self
            .client
            .chat(&req)
            .await
            .map_err(ContextError::Summarization)?;

        let text = resp.text().unwrap_or("").to_string();

        // I-1: Reject empty or whitespace-only summaries.
        if text.trim().is_empty() {
            return Err(ContextError::EmptySummary(
                "LLM returned an empty or whitespace-only summary".to_string(),
            ));
        }

        // Return raw summary text WITHOUT the prefix.
        // The caller (compressor) is responsible for prepending SUMMARY_PREFIX
        // when building the final user message.
        Ok(text)
    }
}

/// Format middle-zone messages into readable text for the summarization prompt.
#[must_use]
pub fn format_turns_for_summary(messages: &[ChatMessage]) -> String {
    let mut buf = String::new();
    for (i, msg) in messages.iter().enumerate() {
        if i > 0 {
            buf.push_str("\n---\n");
        }
        buf.push_str(&extract_text_from_message(msg));
    }
    buf
}

/// Build an iterative summarization prompt that incorporates a previous summary.
///
/// The `previous_summary` should be raw text (without `SUMMARY_PREFIX`).
/// If it still contains the prefix, it is automatically stripped.
///
/// The iterative template instructs the LLM to carry forward Key Decisions
/// and Critical Context sections verbatim.
#[must_use]
pub fn build_iterative_prompt(previous_summary: &str, new_turns: &str) -> String {
    // W-4: Strip SUMMARY_PREFIX if present (defensive).
    let clean_summary = previous_summary
        .strip_prefix(SUMMARY_PREFIX)
        .map_or(previous_summary, str::trim_start);

    #[allow(clippy::literal_string_with_formatting_args)]
    ITERATIVE_USER_TEMPLATE
        .replace("{previous_summary}", clean_summary)
        .replace("{new_turns}", new_turns)
}
