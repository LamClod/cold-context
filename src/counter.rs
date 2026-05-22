use cold_sdk::{ChatMessage, ContentPart, MessageContent};

/// Trait for estimating token counts of messages.
pub trait TokenCounter: Send + Sync {
    /// Estimate tokens for a single message.
    fn count_message(&self, message: &ChatMessage) -> u32;

    /// Estimate tokens for a slice of messages.
    fn count_messages(&self, messages: &[ChatMessage]) -> u32 {
        messages.iter().map(|m| self.count_message(m)).sum()
    }
}

/// Simple character-based token estimator (~3 chars per token).
///
/// Uses a conservative ratio of 3 chars per token, which is closer to real
/// tokenizer behavior for mixed content and safer (overestimates slightly,
/// so compression triggers earlier rather than risking `context_length_exceeded`).
#[derive(Debug, Clone, Copy, Default)]
pub struct CharEstimator;

impl CharEstimator {
    /// Flat token cost for an image URL part.
    const IMAGE_TOKENS: u32 = 1600;
    /// Per-message overhead (role, delimiters, etc.).
    const MESSAGE_OVERHEAD: u32 = 4;
    /// Chars per token ratio (conservative: slightly overestimates tokens).
    const CHARS_PER_TOKEN: u32 = 3;

    fn count_content(content: &MessageContent) -> u32 {
        match content {
            MessageContent::Text(t) => Self::chars_to_tokens(t.len()),
            MessageContent::Parts(parts) => parts.iter().map(Self::count_part).sum(),
        }
    }

    fn count_part(part: &ContentPart) -> u32 {
        match part {
            ContentPart::Text { text } => Self::chars_to_tokens(text.len()),
            ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => Self::IMAGE_TOKENS,
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    const fn chars_to_tokens(chars: usize) -> u32 {
        // Saturate instead of silently truncating usize to u32.
        let result = chars / Self::CHARS_PER_TOKEN as usize;
        if result > u32::MAX as usize {
            u32::MAX
        } else {
            result as u32
        }
    }
}

impl TokenCounter for CharEstimator {
    fn count_message(&self, message: &ChatMessage) -> u32 {
        let mut tokens = Self::MESSAGE_OVERHEAD;

        if let Some(ref content) = message.content {
            tokens += Self::count_content(content);
        }

        if let Some(ref tool_calls) = message.tool_calls {
            for tc in tool_calls {
                tokens += Self::chars_to_tokens(tc.function.arguments.len());
                tokens += Self::chars_to_tokens(tc.function.name.len());
                tokens += Self::chars_to_tokens(tc.id.len());
            }
        }

        tokens
    }
}
