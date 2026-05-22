use cold_sdk::{ChatMessage, ContentPart, MessageContent, Role};

/// Find the largest byte position <= `max_bytes` that lies on a UTF-8 char boundary.
///
/// This prevents panics when slicing multi-byte strings at an arbitrary byte offset.
#[must_use]
pub fn safe_truncate_pos(s: &str, max_bytes: usize) -> usize {
    if max_bytes >= s.len() {
        return s.len();
    }
    let mut pos = max_bytes;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Multiplier to convert image tokens (1600) back to char-equivalent length.
const IMAGE_CHAR_EQUIVALENT: usize = 4800; // 1600 tokens * 3 chars/token

/// Total text character length of message content. Images count as 6400 chars.
#[must_use]
pub fn content_text_length(content: Option<&MessageContent>) -> usize {
    let Some(content) = content else { return 0 };
    match content {
        MessageContent::Text(t) => t.len(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => text.len(),
                // W-11: Use a flat estimate for audio (like images) instead of
                // counting base64 chars which vastly over-counts.
                ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => {
                    IMAGE_CHAR_EQUIVALENT
                }
            })
            .sum(),
    }
}

/// Best-effort text extraction from optional message content.
#[must_use]
pub fn extract_text(content: Option<&MessageContent>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    match content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Parts(parts) => {
            let mut buf = String::new();
            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str(text);
                    }
                    ContentPart::ImageUrl { .. } => {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str("[image]");
                    }
                    ContentPart::InputAudio { .. } => {
                        if !buf.is_empty() {
                            buf.push('\n');
                        }
                        buf.push_str("[audio]");
                    }
                }
            }
            buf
        }
    }
}

/// Extract a text representation of a message including role, content, and tool calls.
#[must_use]
pub fn extract_text_from_message(msg: &ChatMessage) -> String {
    let mut buf = String::new();

    let role_str = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
        Role::Developer => "developer",
        Role::Unknown => "unknown",
    };
    buf.push_str(role_str);
    buf.push_str(": ");

    let content_text = extract_text(msg.content.as_ref());
    buf.push_str(&content_text);

    if let Some(ref tool_calls) = msg.tool_calls {
        for tc in tool_calls {
            // W-2: Include tool_call id so linkage to tool results is preserved.
            buf.push_str("\n  [tool_call id=");
            buf.push_str(&tc.id);
            buf.push_str("] ");
            buf.push_str(&tc.function.name);
            buf.push('(');
            // Truncate long arguments for readability
            let args = &tc.function.arguments;
            if args.len() > 200 {
                buf.push_str(&args[..safe_truncate_pos(args, 200)]);
                buf.push_str("...");
            } else {
                buf.push_str(args);
            }
            buf.push(')');
        }
    }

    if let Some(ref tool_call_id) = msg.tool_call_id {
        buf.push_str(" [tool_call_id=");
        buf.push_str(tool_call_id);
        buf.push(']');
    }

    buf
}
