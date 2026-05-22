//! Post-compact file restoration.
//!
//! After LLM summarization, re-inject content of recently-read files so the
//! model does not lose working context.

use std::path::Path;

use cold_sdk::{ChatMessage, Role};

/// Configuration for post-compact file restoration.
#[derive(Debug, Clone)]
pub struct RestoreConfig {
    /// Max files to restore (default 5).
    pub max_files: usize,
    /// Max tokens per file (default 5000).
    pub max_tokens_per_file: u32,
    /// Total token budget for restoration (default 25000).
    pub total_token_budget: u32,
}

impl Default for RestoreConfig {
    fn default() -> Self {
        Self {
            max_files: 5,
            max_tokens_per_file: 5000,
            total_token_budget: 25_000,
        }
    }
}

/// Scan messages for recent `read_file` tool calls and collect file paths.
///
/// Returns the most recent `max_files` unique file paths.
#[must_use]
pub fn find_recent_reads(messages: &[ChatMessage], max_files: usize) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Scan in reverse for recency
    for msg in messages.iter().rev() {
        if msg.role != Role::Assistant {
            continue;
        }
        let Some(ref tool_calls) = msg.tool_calls else {
            continue;
        };
        for tc in tool_calls {
            if !is_read_file_tool(&tc.function.name) {
                continue;
            }
            // Extract "path" or "file_path" from JSON arguments
            if let Some(path) = extract_path_arg(&tc.function.arguments) {
                if seen.insert(path.clone()) {
                    paths.push(path);
                    if paths.len() >= max_files {
                        return paths;
                    }
                }
            }
        }
    }

    paths
}

/// Build restoration messages from file paths.
///
/// Reads files from disk and creates user messages with their content.
pub async fn build_restoration_messages(
    paths: &[String],
    config: &RestoreConfig,
    root: &Path,
) -> Vec<ChatMessage> {
    const CHARS_PER_TOKEN: u32 = 3;

    let mut result = Vec::new();
    let mut total_tokens: u32 = 0;

    for path_str in paths {
        let file_path = if Path::new(path_str).is_absolute() {
            std::path::PathBuf::from(path_str)
        } else {
            root.join(path_str)
        };

        let Ok(content) = tokio::fs::read_to_string(&file_path).await else {
            continue;
        };

        #[allow(clippy::cast_possible_truncation)]
        let file_tokens = (content.len() as u32) / CHARS_PER_TOKEN;

        if file_tokens > config.max_tokens_per_file {
            // Truncate to fit
            let max_chars = (config.max_tokens_per_file * CHARS_PER_TOKEN) as usize;
            let pos = crate::util::safe_truncate_pos(&content, max_chars);
            let truncated = &content[..pos];
            let msg_text = format!(
                "[File restored after context compaction]\n\nFile: {path_str}\n```\n{truncated}\n...[truncated]\n```"
            );
            #[allow(clippy::cast_possible_truncation)]
            let msg_tokens = (msg_text.len() as u32) / CHARS_PER_TOKEN;
            if total_tokens + msg_tokens > config.total_token_budget {
                break;
            }
            total_tokens += msg_tokens;
            result.push(ChatMessage::user(msg_text));
        } else {
            let msg_text = format!(
                "[File restored after context compaction]\n\nFile: {path_str}\n```\n{content}\n```"
            );
            #[allow(clippy::cast_possible_truncation)]
            let msg_tokens = (msg_text.len() as u32) / CHARS_PER_TOKEN;
            if total_tokens + msg_tokens > config.total_token_budget {
                break;
            }
            total_tokens += msg_tokens;
            result.push(ChatMessage::user(msg_text));
        }
    }

    result
}

/// Check if a tool name is a read-file variant.
fn is_read_file_tool(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "read_file" | "read" | "readfile"
    )
}

/// Extract the file path from tool call arguments JSON.
fn extract_path_arg(arguments: &str) -> Option<String> {
    let val: serde_json::Value = serde_json::from_str(arguments).ok()?;
    // Try common field names
    for key in &["path", "file_path", "filePath", "filename"] {
        if let Some(serde_json::Value::String(s)) = val.get(*key) {
            if !s.is_empty() {
                return Some(s.clone());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use cold_sdk::{FunctionCall, ToolCall};

    use super::*;

    fn make_tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
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
    fn find_recent_reads_extracts_paths() {
        let msgs = vec![
            ChatMessage::user("read foo"),
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc1",
                "read_file",
                r#"{"path":"src/main.rs"}"#,
            )]),
            ChatMessage::tool("tc1", "fn main() {}"),
            ChatMessage::user("read bar"),
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc2",
                "Read",
                r#"{"file_path":"src/lib.rs"}"#,
            )]),
            ChatMessage::tool("tc2", "pub mod foo;"),
        ];

        let paths = find_recent_reads(&msgs, 5);
        assert_eq!(paths.len(), 2);
        // Most recent first
        assert_eq!(paths[0], "src/lib.rs");
        assert_eq!(paths[1], "src/main.rs");
    }

    #[test]
    fn find_recent_reads_deduplicates() {
        let msgs = vec![
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc1",
                "read_file",
                r#"{"path":"src/main.rs"}"#,
            )]),
            ChatMessage::tool("tc1", "v1"),
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc2",
                "read_file",
                r#"{"path":"src/main.rs"}"#,
            )]),
            ChatMessage::tool("tc2", "v2"),
        ];

        let paths = find_recent_reads(&msgs, 5);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], "src/main.rs");
    }

    #[test]
    fn find_recent_reads_respects_max() {
        let msgs = vec![
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc1",
                "read_file",
                r#"{"path":"a.rs"}"#,
            )]),
            ChatMessage::tool("tc1", "a"),
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc2",
                "read_file",
                r#"{"path":"b.rs"}"#,
            )]),
            ChatMessage::tool("tc2", "b"),
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc3",
                "read_file",
                r#"{"path":"c.rs"}"#,
            )]),
            ChatMessage::tool("tc3", "c"),
        ];

        let paths = find_recent_reads(&msgs, 2);
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn find_recent_reads_ignores_non_read_tools() {
        let msgs = vec![
            make_assistant_with_tool_calls(vec![make_tool_call(
                "tc1",
                "bash",
                r#"{"command":"ls"}"#,
            )]),
            ChatMessage::tool("tc1", "file1.rs"),
        ];

        let paths = find_recent_reads(&msgs, 5);
        assert!(paths.is_empty());
    }

    #[test]
    fn extract_path_arg_handles_variants() {
        assert_eq!(
            extract_path_arg(r#"{"path":"foo.rs"}"#),
            Some("foo.rs".to_string())
        );
        assert_eq!(
            extract_path_arg(r#"{"file_path":"bar.rs"}"#),
            Some("bar.rs".to_string())
        );
        assert_eq!(
            extract_path_arg(r#"{"filePath":"baz.rs"}"#),
            Some("baz.rs".to_string())
        );
        assert_eq!(extract_path_arg(r#"{"other":"val"}"#), None);
        assert_eq!(extract_path_arg("not json"), None);
    }

    #[tokio::test]
    async fn build_restoration_messages_empty_paths() {
        let config = RestoreConfig::default();
        let msgs = build_restoration_messages(&[], &config, Path::new("/tmp")).await;
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn build_restoration_messages_nonexistent_file() {
        let config = RestoreConfig::default();
        let paths = vec!["/nonexistent/path/to/file.rs".to_string()];
        let msgs = build_restoration_messages(&paths, &config, Path::new("/tmp")).await;
        assert!(msgs.is_empty());
    }
}
