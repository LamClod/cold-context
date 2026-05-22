use cold_context::integrity::{IntegrityIssue, repair_sequence, validate_sequence};
use cold_sdk::{ChatMessage, FunctionCall, Role, ToolCall};

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
fn valid_sequence_no_issues() {
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("hi"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read_file", "{}")]),
        ChatMessage::tool("tc1", "content"),
        ChatMessage::assistant("done"),
    ];

    let issues = validate_sequence(&msgs);
    assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
}

#[test]
fn orphaned_tool_result_detected_and_repaired() {
    // Tool message references a tool_call_id that no assistant has
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("hi"),
        ChatMessage::tool("nonexistent_tc", "some output"),
        ChatMessage::assistant("ok"),
    ];

    let issues = validate_sequence(&msgs);
    assert!(
        issues.iter().any(|i| matches!(i, IntegrityIssue::OrphanedToolResult { tool_call_id, .. } if tool_call_id == "nonexistent_tc")),
        "expected OrphanedToolResult, got: {issues:?}"
    );

    // Repair should remove the orphaned tool message
    let repaired = repair_sequence(msgs);
    assert_eq!(repaired.len(), 3); // system, user, assistant
    assert!(!repaired.iter().any(|m| m.role == Role::Tool));
}

#[test]
fn orphaned_tool_use_detected() {
    // Assistant has tool_calls but no matching tool results
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("hi"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc_orphan", "search", "{}")]),
        ChatMessage::user("next question"),
    ];

    let issues = validate_sequence(&msgs);
    assert!(
        issues.iter().any(|i| matches!(i, IntegrityIssue::OrphanedToolUse { missing_ids, .. } if missing_ids.contains(&"tc_orphan".to_string()))),
        "expected OrphanedToolUse, got: {issues:?}"
    );
}

#[test]
fn tool_message_with_none_tool_call_id_removed_by_repair() {
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("hi"),
        // Tool message with tool_call_id: None
        ChatMessage {
            role: Role::Tool,
            content: Some(cold_sdk::MessageContent::Text("orphan".to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
            refusal: None,
        },
        ChatMessage::assistant("ok"),
    ];

    let repaired = repair_sequence(msgs);
    // The tool message with None id should be removed
    assert_eq!(repaired.len(), 3);
    assert!(!repaired.iter().any(|m| m.role == Role::Tool));
}

#[test]
fn mixed_valid_and_invalid_only_invalid_removed() {
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("hi"),
        make_assistant_with_tool_calls(vec![
            make_tool_call("tc_valid", "read", "{}"),
            make_tool_call("tc_no_result", "write", "{}"),
        ]),
        ChatMessage::tool("tc_valid", "file content"),
        // Missing tc_no_result tool message
        // Plus an orphaned tool result with unknown id
        ChatMessage::tool("unknown_id", "garbage"),
        ChatMessage::assistant("done"),
    ];

    let issues = validate_sequence(&msgs);
    // Should find orphaned tool result (unknown_id) and orphaned tool use (tc_no_result)
    assert!(issues.len() >= 2, "expected >=2 issues, got: {issues:?}");

    let repaired = repair_sequence(msgs);
    // "unknown_id" tool message removed; tc_no_result removed from tool_calls
    assert!(
        !repaired
            .iter()
            .any(|m| { m.role == Role::Tool && m.tool_call_id.as_deref() == Some("unknown_id") })
    );
    // The assistant's tool_calls should only have tc_valid now
    let assistant = repaired.iter().find(|m| m.tool_calls.is_some()).unwrap();
    let tcs = assistant.tool_calls.as_ref().unwrap();
    assert_eq!(tcs.len(), 1);
    assert_eq!(tcs[0].id, "tc_valid");
    // Valid tool result preserved
    assert!(
        repaired
            .iter()
            .any(|m| { m.role == Role::Tool && m.tool_call_id.as_deref() == Some("tc_valid") })
    );
}
