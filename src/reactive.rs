//! Reactive compact: emergency context recovery after `context_length_exceeded`.
//!
//! When the API returns a "prompt too long" error, this module drops the oldest
//! API-round groups until the estimated token count is below `target_tokens`.
//! This is a fast, no-LLM operation for emergency recovery.

use std::ops::Range;

use cold_sdk::{ChatMessage, Role};

use crate::counter::{CharEstimator, TokenCounter};
use crate::error::ContextError;

/// Result of a reactive compact operation.
#[derive(Debug)]
pub struct ReactiveCompactResult {
    /// The compacted messages.
    pub messages: Vec<ChatMessage>,
    /// How many groups were dropped from the head.
    pub groups_dropped: usize,
    /// Estimated tokens after compaction.
    pub estimated_tokens: u32,
}

/// Group messages by API round (user -> assistant -> tool* -> assistant -> ...).
///
/// Each group is one logical exchange:
/// - A group starts with a `User` message (or the first `System` message for the
///   initial group).
/// - A group includes all following messages until the next `User` message.
/// - `Tool` messages belong to the same group as their preceding `Assistant` message.
/// - Leading `System` messages form their own group (always protected).
#[must_use]
pub fn group_by_api_round(messages: &[ChatMessage]) -> Vec<Range<usize>> {
    if messages.is_empty() {
        return Vec::new();
    }

    let mut groups: Vec<Range<usize>> = Vec::new();
    let mut current_start: usize = 0;

    // Collect leading system messages into group 0.
    let mut i = 0;
    while i < messages.len() && messages[i].role == Role::System {
        i += 1;
    }

    if i > 0 {
        // There are leading system messages — they form their own group.
        groups.push(0..i);
        current_start = i;
    }

    // Walk the rest: each User message starts a new group.
    let mut group_start = current_start;
    for (idx, msg) in messages.iter().enumerate().skip(current_start) {
        if msg.role == Role::User && idx > group_start {
            groups.push(group_start..idx);
            group_start = idx;
        }
    }

    // Push the final group.
    if group_start < messages.len() {
        groups.push(group_start..messages.len());
    }

    groups
}

/// Perform reactive compaction by dropping oldest API-round groups until the
/// estimated token count is below `target_tokens`.
///
/// Strategy:
/// 1. Group messages by API round.
/// 2. Always protect: first group (system / initial exchange) and last
///    `protect_last_n_groups` groups.
/// 3. Drop oldest unprotected groups one at a time.
/// 4. After each drop, re-estimate tokens.
/// 5. Stop when under target or no more groups to drop.
///
/// # Errors
///
/// Returns [`ContextError::ReactiveCompactFailed`] if all droppable groups have
/// been removed but the estimated token count is still above `target_tokens`.
pub fn reactive_compact(
    messages: Vec<ChatMessage>,
    target_tokens: u32,
    protect_last_n_groups: usize,
) -> Result<ReactiveCompactResult, ContextError> {
    let counter = CharEstimator;
    let current_tokens = counter.count_messages(&messages);

    // Already under target — nothing to do.
    if current_tokens <= target_tokens {
        return Ok(ReactiveCompactResult {
            messages,
            groups_dropped: 0,
            estimated_tokens: current_tokens,
        });
    }

    let groups = group_by_api_round(&messages);

    if groups.is_empty() {
        return Err(ContextError::ReactiveCompactFailed(
            "no groups to drop".to_string(),
        ));
    }

    // Determine protected ranges:
    //   - group 0 is always protected (system / initial exchange)
    //   - last N groups are protected
    let total_groups = groups.len();
    let tail_protected_start = total_groups.saturating_sub(protect_last_n_groups);
    // The droppable range is groups[1..tail_protected_start] (indices 1..tail_protected_start).
    let droppable_end = tail_protected_start.max(1);

    // Collect which groups to keep. Start by trying to drop from index 1 upwards.
    let mut drop_up_to: usize = 1; // exclusive: groups[1..drop_up_to] are dropped
    let mut groups_dropped: usize = 0;
    let mut estimated = current_tokens;

    for (group_idx, group) in groups.iter().enumerate().take(droppable_end).skip(1) {
        let group_tokens: u32 = messages[group.clone()]
            .iter()
            .map(|m| counter.count_message(m))
            .sum();
        estimated = estimated.saturating_sub(group_tokens);
        drop_up_to = group_idx + 1;
        groups_dropped += 1;

        if estimated <= target_tokens {
            break;
        }
    }

    // Build the result message list: keep group 0 + groups[drop_up_to..].
    let mut result: Vec<ChatMessage> = Vec::new();

    // Group 0 (always kept).
    let first_range = &groups[0];
    for msg in &messages[first_range.clone()] {
        result.push(msg.clone());
    }

    // Remaining kept groups.
    for group in &groups[drop_up_to..] {
        for msg in &messages[group.clone()] {
            result.push(msg.clone());
        }
    }

    let final_tokens = counter.count_messages(&result);

    if final_tokens > target_tokens {
        return Err(ContextError::ReactiveCompactFailed(format!(
            "still at {final_tokens} tokens after dropping {groups_dropped} group(s) \
             (target: {target_tokens})"
        )));
    }

    Ok(ReactiveCompactResult {
        messages: result,
        groups_dropped,
        estimated_tokens: final_tokens,
    })
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

    // ---------------------------------------------------------------
    // group_by_api_round tests
    // ---------------------------------------------------------------

    #[test]
    fn group_empty() {
        let groups = group_by_api_round(&[]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_system_only() {
        let msgs = vec![ChatMessage::system("sys")];
        let groups = group_by_api_round(&msgs);
        assert_eq!(groups, vec![0..1]);
    }

    #[test]
    fn group_simple_conversation() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
            ChatMessage::user("bye"),
            ChatMessage::assistant("goodbye"),
        ];
        let groups = group_by_api_round(&msgs);
        assert_eq!(groups, vec![0..1, 1..3, 3..5]);
    }

    #[test]
    fn group_with_tool_messages() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("read file"),
            make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read_file")]),
            ChatMessage::tool("tc1", "file content"),
            ChatMessage::assistant("here is the file"),
            ChatMessage::user("thanks"),
            ChatMessage::assistant("you're welcome"),
        ];
        let groups = group_by_api_round(&msgs);
        // Group 0: system
        // Group 1: user + assistant(tool_call) + tool + assistant
        // Group 2: user + assistant
        assert_eq!(groups, vec![0..1, 1..5, 5..7]);
    }

    #[test]
    fn group_no_system_messages() {
        let msgs = vec![
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
            ChatMessage::user("bye"),
            ChatMessage::assistant("goodbye"),
        ];
        let groups = group_by_api_round(&msgs);
        assert_eq!(groups, vec![0..2, 2..4]);
    }

    #[test]
    fn group_multiple_system_messages() {
        let msgs = vec![
            ChatMessage::system("sys1"),
            ChatMessage::system("sys2"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];
        let groups = group_by_api_round(&msgs);
        assert_eq!(groups, vec![0..2, 2..4]);
    }

    // ---------------------------------------------------------------
    // reactive_compact tests
    // ---------------------------------------------------------------

    #[test]
    fn compact_already_under_target() {
        let msgs = vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let result = reactive_compact(msgs.clone(), 10_000, 3).unwrap();
        assert_eq!(result.groups_dropped, 0);
        assert_eq!(result.messages.len(), msgs.len());
    }

    #[test]
    fn compact_drops_oldest_groups() {
        // Build a conversation with many groups, each containing a large message.
        let large = "x".repeat(3000); // ~1000 tokens per message
        let mut msgs = vec![ChatMessage::system("sys")];
        for i in 0..10 {
            msgs.push(ChatMessage::user(format!("q{i}: {large}")));
            msgs.push(ChatMessage::assistant(format!("a{i}: {large}")));
        }

        let counter = CharEstimator;
        let total = counter.count_messages(&msgs);

        // Target is about half — should drop some groups.
        let target = total / 2;
        let result = reactive_compact(msgs, target, 3).unwrap();

        assert!(result.groups_dropped > 0);
        assert!(result.estimated_tokens <= target);
        // System message should still be first.
        assert_eq!(result.messages[0].role, Role::System);
    }

    #[test]
    fn compact_protects_last_n_groups() {
        let large = "x".repeat(3000);
        let mut msgs = vec![ChatMessage::system("sys")];
        for i in 0..6 {
            msgs.push(ChatMessage::user(format!("q{i}: {large}")));
            msgs.push(ChatMessage::assistant(format!("a{i}: {large}")));
        }

        // Very low target — should drop as many as possible but keep last 3.
        let result = reactive_compact(msgs, 100, 3);

        // May fail because even after dropping everything droppable, it's too large.
        // But the last 3 groups should always be present.
        match result {
            Ok(r) => {
                // The last 3 user messages should be q3, q4, q5
                let user_msgs: Vec<_> = r
                    .messages
                    .iter()
                    .filter(|m| m.role == Role::User)
                    .collect();
                // At minimum, the last 3 groups' user messages should be present.
                assert!(user_msgs.len() >= 3);
            }
            Err(ContextError::ReactiveCompactFailed(_)) => {
                // Expected when target is impossibly low.
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn compact_fails_when_impossible() {
        // Single huge system message that exceeds target on its own.
        let huge = "x".repeat(30_000);
        let msgs = vec![
            ChatMessage::system(&huge),
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello"),
        ];
        let result = reactive_compact(msgs, 10, 3);
        assert!(result.is_err());
    }

    #[test]
    fn compact_with_tool_groups() {
        let large = "x".repeat(3000);
        let mut msgs = vec![ChatMessage::system("sys")];

        // Group 1: user + assistant(tool) + tool + assistant
        msgs.push(ChatMessage::user(format!("read: {large}")));
        msgs.push(make_assistant_with_tool_calls(vec![make_tool_call(
            "tc1",
            "read_file",
        )]));
        msgs.push(ChatMessage::tool("tc1", &large));
        msgs.push(ChatMessage::assistant(format!("result: {large}")));

        // Group 2: simple exchange
        msgs.push(ChatMessage::user(format!("q2: {large}")));
        msgs.push(ChatMessage::assistant(format!("a2: {large}")));

        // Group 3: simple exchange (protected tail)
        msgs.push(ChatMessage::user("final question"));
        msgs.push(ChatMessage::assistant("final answer"));

        let counter = CharEstimator;
        let total = counter.count_messages(&msgs);
        let target = total / 2;

        let result = reactive_compact(msgs, target, 1).unwrap();
        assert!(result.groups_dropped > 0);
        assert!(result.estimated_tokens <= target);
        // System message preserved.
        assert_eq!(result.messages[0].role, Role::System);
    }
}
