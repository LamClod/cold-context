//! Microcompact: time-based tool result clearing.
//!
//! Before the main compression pipeline runs, clear old tool result content
//! that is no longer relevant. This is a free operation (no LLM call) that
//! can significantly reduce token count.

use std::collections::HashMap;

use cold_sdk::{ChatMessage, MessageContent, Role};

use crate::util::content_text_length;

/// Tool names whose results are safe to clear when stale.
const CLEARABLE_TOOLS: &[&str] = &[
    "read_file",
    "search_files",
    "list_dir",
    "terminal",
    "bash",
    "grep",
    "glob",
    "Read",
    "Grep",
    "Glob",
    "Bash",
    "ListDir",
];

/// Tool names whose results should never be cleared.
const PROTECTED_TOOLS: &[&str] = &[
    "think",
    "clarify",
    "memory",
    "TodoRead",
    "TodoWrite",
    "memory_read",
    "memory_write",
];

/// Configuration for the microcompact pass.
#[derive(Debug, Clone)]
pub struct MicrocompactConfig {
    /// Clear tool results older than this many turns from the tail (default 10).
    pub stale_turn_threshold: usize,
    /// Only clear results larger than this many bytes (default 200).
    pub min_content_bytes: usize,
    /// Replacement text.
    pub placeholder: &'static str,
}

impl Default for MicrocompactConfig {
    fn default() -> Self {
        Self {
            stale_turn_threshold: 10,
            min_content_bytes: 200,
            placeholder: "[Old tool result cleared to save context]",
        }
    }
}

/// Apply microcompact to messages.
///
/// Returns `(new_messages, cleared_count, estimated_tokens_saved)`.
#[must_use]
pub fn microcompact(
    messages: Vec<ChatMessage>,
    config: &MicrocompactConfig,
) -> (Vec<ChatMessage>, usize, u32) {
    const CHARS_PER_TOKEN: usize = 3;

    let len = messages.len();
    let tail_boundary = len.saturating_sub(config.stale_turn_threshold);

    if tail_boundary == 0 {
        return (messages, 0, 0);
    }

    // Build lookup: tool_call_id -> tool_name from assistant messages
    let mut tool_name_lookup: HashMap<String, String> = HashMap::new();
    for msg in &messages {
        if msg.role == Role::Assistant {
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    tool_name_lookup.insert(tc.id.clone(), tc.function.name.clone());
                }
            }
        }
    }

    let placeholder_len = config.placeholder.len();
    let mut result = Vec::with_capacity(len);
    let mut cleared: usize = 0;
    let mut tokens_saved: u32 = 0;

    for (i, msg) in messages.into_iter().enumerate() {
        if i < tail_boundary && msg.role == Role::Tool {
            let content_len = content_text_length(msg.content.as_ref());
            if content_len > config.min_content_bytes {
                // Determine tool name
                let tool_name = msg
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| tool_name_lookup.get(id).map(String::as_str));

                let should_clear = tool_name.is_some_and(|name| {
                    // Explicitly protected tools are never cleared
                    if PROTECTED_TOOLS.iter().any(|p| name.eq_ignore_ascii_case(p)) {
                        false
                    } else {
                        // Clear if it's a known clearable tool
                        CLEARABLE_TOOLS.iter().any(|c| name.eq_ignore_ascii_case(c))
                    }
                });

                if should_clear {
                    #[allow(clippy::cast_possible_truncation)]
                    let delta =
                        (content_len.saturating_sub(placeholder_len) / CHARS_PER_TOKEN) as u32;
                    tokens_saved += delta;
                    cleared += 1;

                    let mut new_msg = msg;
                    new_msg.content =
                        Some(MessageContent::Text(config.placeholder.to_string()));
                    result.push(new_msg);
                    continue;
                }
            }
        }
        result.push(msg);
    }

    (result, cleared, tokens_saved)
}

#[cfg(test)]
mod tests {
    use cold_sdk::{FunctionCall, ToolCall};

    use super::*;

    fn make_tool_call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    fn make_assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            content: None,
            name: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            refusal: None,
        }
    }

    #[test]
    fn clears_old_read_file_results() {
        let large_content = "x".repeat(500);
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("read file foo"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read_file")]),
            ChatMessage::tool("tc1", &large_content),
            ChatMessage::user("read file bar"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc2", "read_file")]),
            ChatMessage::tool("tc2", &large_content),
        ];
        // Add tail messages to push old ones before boundary
        for i in 0..12 {
            msgs.push(ChatMessage::user(format!("q{i}")));
            msgs.push(ChatMessage::assistant(format!("a{i}")));
        }

        let config = MicrocompactConfig::default();
        let (result, cleared, saved) = microcompact(msgs, &config);

        assert_eq!(cleared, 2);
        assert!(saved > 0);
        // The tool results should be replaced
        assert!(result.iter().any(|m| {
            if let Some(MessageContent::Text(t)) = &m.content {
                t == config.placeholder
            } else {
                false
            }
        }));
    }

    #[test]
    fn does_not_clear_think_results() {
        let large_content = "x".repeat(500);
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("think about this"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc1", "think")]),
            ChatMessage::tool("tc1", &large_content),
        ];
        for i in 0..12 {
            msgs.push(ChatMessage::user(format!("q{i}")));
            msgs.push(ChatMessage::assistant(format!("a{i}")));
        }

        let config = MicrocompactConfig::default();
        let (_, cleared, _) = microcompact(msgs, &config);

        assert_eq!(cleared, 0);
    }

    #[test]
    fn does_not_clear_recent_messages() {
        let large_content = "x".repeat(500);
        // Only a few messages total — everything is in the tail
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("read file"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read_file")]),
            ChatMessage::tool("tc1", &large_content),
            ChatMessage::user("done"),
            ChatMessage::assistant("ok"),
        ];

        let config = MicrocompactConfig::default();
        let (_, cleared, _) = microcompact(msgs, &config);

        assert_eq!(cleared, 0);
    }

    #[test]
    fn does_not_clear_small_results() {
        let small_content = "ok";
        let mut msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("read file"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read_file")]),
            ChatMessage::tool("tc1", small_content),
        ];
        for i in 0..12 {
            msgs.push(ChatMessage::user(format!("q{i}")));
            msgs.push(ChatMessage::assistant(format!("a{i}")));
        }

        let config = MicrocompactConfig::default();
        let (_, cleared, _) = microcompact(msgs, &config);

        assert_eq!(cleared, 0);
    }

    #[test]
    fn empty_messages_returns_empty() {
        let config = MicrocompactConfig::default();
        let (result, cleared, saved) = microcompact(vec![], &config);
        assert!(result.is_empty());
        assert_eq!(cleared, 0);
        assert_eq!(saved, 0);
    }
}
