use std::collections::HashSet;
use std::fmt;

use cold_sdk::{ChatMessage, Role};

/// An issue found during sequence validation.
#[derive(Debug, Clone)]
pub enum IntegrityIssue {
    /// A tool message references a `tool_call_id` that no assistant message has.
    OrphanedToolResult { index: usize, tool_call_id: String },
    /// An assistant message has `tool_calls` but no matching tool results follow.
    OrphanedToolUse {
        index: usize,
        missing_ids: Vec<String>,
    },
    /// A tool message appears where one is not expected (no preceding assistant with `tool_calls`).
    UnexpectedToolMessage { index: usize },
}

impl fmt::Display for IntegrityIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OrphanedToolResult {
                index,
                tool_call_id,
            } => write!(
                f,
                "orphaned tool result at index {index} (tool_call_id={tool_call_id})"
            ),
            Self::OrphanedToolUse { index, missing_ids } => write!(
                f,
                "orphaned tool use at index {index} (missing results for: {})",
                missing_ids.join(", ")
            ),
            Self::UnexpectedToolMessage { index } => {
                write!(f, "unexpected tool message at index {index}")
            }
        }
    }
}

/// Validate the message sequence for integrity issues.
#[must_use]
pub fn validate_sequence(messages: &[ChatMessage]) -> Vec<IntegrityIssue> {
    let mut issues = Vec::new();

    // Collect all tool_call IDs from assistant messages
    let mut all_tool_call_ids: HashSet<&str> = HashSet::new();
    for msg in messages {
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                all_tool_call_ids.insert(&tc.id);
            }
        }
    }

    // Collect all answered tool_call_ids from tool messages
    let mut answered_ids: HashSet<&str> = HashSet::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::Tool {
            if let Some(ref id) = msg.tool_call_id {
                if all_tool_call_ids.contains(id.as_str()) {
                    answered_ids.insert(id);
                } else {
                    issues.push(IntegrityIssue::OrphanedToolResult {
                        index: i,
                        tool_call_id: id.clone(),
                    });
                }
            }
        }
    }

    // Check for assistant tool_calls with no matching tool results
    for (i, msg) in messages.iter().enumerate() {
        if let Some(ref tool_calls) = msg.tool_calls {
            let missing: Vec<String> = tool_calls
                .iter()
                .filter(|tc| !answered_ids.contains(tc.id.as_str()))
                .map(|tc| tc.id.clone())
                .collect();
            if !missing.is_empty() {
                issues.push(IntegrityIssue::OrphanedToolUse {
                    index: i,
                    missing_ids: missing,
                });
            }
        }
    }

    // Check for unexpected tool messages (not preceded by an assistant with tool_calls)
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::Tool && i > 0 {
            // Walk backward to find the assistant that should own this tool result
            let mut found_owner = false;
            for j in (0..i).rev() {
                if messages[j].role == Role::Assistant && messages[j].tool_calls.is_some() {
                    found_owner = true;
                    break;
                }
                // If we encounter user/system before finding an assistant with tool_calls, it's unexpected
                if messages[j].role == Role::User || messages[j].role == Role::System {
                    break;
                }
            }
            if !found_owner {
                issues.push(IntegrityIssue::UnexpectedToolMessage { index: i });
            }
        }
    }

    issues
}

/// Remove orphaned messages to repair the sequence.
///
/// - Removes tool messages whose `tool_call_id` references no known assistant `tool_call`.
/// - Removes `tool_calls` from assistant messages that have no matching tool result.
#[must_use]
pub fn repair_sequence(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    // Collect all tool_call IDs from assistant messages
    let mut all_tool_call_ids: HashSet<String> = HashSet::new();
    for msg in &messages {
        if let Some(ref tool_calls) = msg.tool_calls {
            for tc in tool_calls {
                all_tool_call_ids.insert(tc.id.clone());
            }
        }
    }

    // Collect all answered tool_call_ids from tool messages
    let mut answered_ids: HashSet<String> = HashSet::new();
    for msg in &messages {
        if msg.role == Role::Tool {
            if let Some(ref id) = msg.tool_call_id {
                if all_tool_call_ids.contains(id) {
                    answered_ids.insert(id.clone());
                }
            }
        }
    }

    let mut result = Vec::with_capacity(messages.len());
    for mut msg in messages {
        // W-10: Skip Tool messages with tool_call_id: None (treat as invalid)
        if msg.role == Role::Tool && msg.tool_call_id.is_none() {
            continue;
        }

        // Skip orphaned tool results
        if msg.role == Role::Tool {
            if let Some(ref id) = msg.tool_call_id {
                if !all_tool_call_ids.contains(id) {
                    continue;
                }
            }
        }

        // Clean up assistant tool_calls that have no matching tool result
        // W-9: Consume `msg` by value via `into_iter()` — mutate in place instead of cloning.
        if msg.role == Role::Assistant {
            if let Some(ref tool_calls) = msg.tool_calls {
                let kept: Vec<_> = tool_calls
                    .iter()
                    .filter(|tc| answered_ids.contains(&tc.id))
                    .cloned()
                    .collect();
                if kept.len() != tool_calls.len() {
                    if kept.is_empty() {
                        msg.tool_calls = None;
                    } else {
                        msg.tool_calls = Some(kept);
                    }
                    result.push(msg);
                    continue;
                }
            }
        }

        result.push(msg);
    }

    result
}
