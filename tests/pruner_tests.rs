use cold_context::pruner::{
    dedup_tool_results, prune_tool_outputs, strip_historical_images, truncate_tool_args,
};
use cold_sdk::{ChatMessage, ContentPart, FunctionCall, ImageUrl, MessageContent, Role, ToolCall};

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

// ═══════════════════════════════════════════════════════════════
// strip_historical_images
// ═══════════════════════════════════════════════════════════════

#[test]
fn strip_images_before_tail_keeps_images_in_tail() {
    let msgs = vec![
        // Index 0: has image (before tail)
        ChatMessage::user_multimodal(vec![
            ContentPart::Text {
                text: "look".to_string(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://img1.png".to_string(),
                    detail: None,
                },
            },
        ]),
        // Index 1: text only
        ChatMessage::assistant("I see it"),
        // Index 2: in tail, has image (should be kept)
        ChatMessage::user_multimodal(vec![ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://img2.png".to_string(),
                detail: None,
            },
        }]),
    ];

    let (result, count, _delta) = strip_historical_images(msgs, 2);
    assert_eq!(count, 1);
    // Index 0 image replaced with "[image removed]"
    if let Some(MessageContent::Parts(parts)) = &result[0].content {
        assert!(
            parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text { text } if text == "[image removed]"))
        );
        assert!(
            !parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. }))
        );
    } else {
        panic!("expected Parts content");
    }
    // Index 2 image preserved
    if let Some(MessageContent::Parts(parts)) = &result[2].content {
        assert!(
            parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. }))
        );
    } else {
        panic!("expected image preserved in tail");
    }
}

#[test]
fn strip_images_no_images_no_change() {
    let msgs = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
    let (result, count, _delta) = strip_historical_images(msgs.clone(), 1);
    assert_eq!(count, 0);
    assert_eq!(result.len(), 2);
}

// ═══════════════════════════════════════════════════════════════
// prune_tool_outputs
// ═══════════════════════════════════════════════════════════════

#[test]
fn prune_tool_outputs_large_content_replaced() {
    let large_content = "x".repeat(300); // > 200 threshold
    let msgs = vec![
        ChatMessage::system("sys"),
        make_assistant_with_tool_calls(vec![make_tool_call(
            "tc1",
            "read_file",
            r#"{"path":"a.rs"}"#,
        )]),
        ChatMessage::tool("tc1", &large_content),
        ChatMessage::user("done"),
    ];

    let (result, count, _delta) = prune_tool_outputs(msgs, 1, 3);
    assert_eq!(count, 1);
    // The tool message content should be replaced with summary
    if let Some(MessageContent::Text(t)) = &result[2].content {
        assert!(t.contains("[read_file]"));
        assert!(t.contains("300 bytes"));
    } else {
        panic!("expected text replacement");
    }
}

#[test]
fn prune_tool_outputs_small_content_untouched() {
    let small_content = "short output"; // < 200 threshold
    let msgs = vec![
        ChatMessage::system("sys"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "echo", "{}")]),
        ChatMessage::tool("tc1", small_content),
        ChatMessage::user("done"),
    ];

    let (result, count, _delta) = prune_tool_outputs(msgs, 1, 3);
    assert_eq!(count, 0);
    if let Some(MessageContent::Text(t)) = &result[2].content {
        assert_eq!(t, "short output");
    }
}

#[test]
fn prune_tool_outputs_finds_tool_name_from_assistant() {
    let large_content = "y".repeat(500);
    let msgs = vec![
        make_assistant_with_tool_calls(vec![make_tool_call(
            "tc_abc",
            "search_code",
            r#"{"q":"foo"}"#,
        )]),
        ChatMessage::tool("tc_abc", &large_content),
    ];

    let (result, count, _delta) = prune_tool_outputs(msgs, 0, 2);
    assert_eq!(count, 1);
    if let Some(MessageContent::Text(t)) = &result[1].content {
        assert!(
            t.contains("[search_code]"),
            "should contain tool name, got: {t}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// dedup_tool_results
// ═══════════════════════════════════════════════════════════════

#[test]
fn dedup_identical_content_same_tool() {
    let content = "x".repeat(50); // non-empty
    let msgs = vec![
        make_assistant_with_tool_calls(vec![
            make_tool_call("tc1", "read_file", "{}"),
            make_tool_call("tc2", "read_file", "{}"),
        ]),
        ChatMessage::tool("tc1", &content), // older duplicate
        ChatMessage::tool("tc2", &content), // newer duplicate (kept)
    ];

    let (result, count, _delta) = dedup_tool_results(msgs, 0, 3);
    assert_eq!(count, 1);
    // First tool message (older) should be replaced
    if let Some(MessageContent::Text(t)) = &result[1].content {
        assert!(t.contains("Duplicate"), "older should be replaced: {t}");
    }
    // Second (newer) should keep original content
    if let Some(MessageContent::Text(t)) = &result[2].content {
        assert_eq!(t, &content);
    }
}

#[test]
fn dedup_different_tools_same_content_not_deduped() {
    let content = "same content here";
    let msgs = vec![
        make_assistant_with_tool_calls(vec![
            make_tool_call("tc1", "tool_a", "{}"),
            make_tool_call("tc2", "tool_b", "{}"),
        ]),
        ChatMessage::tool("tc1", content), // tool_a
        ChatMessage::tool("tc2", content), // tool_b — different tool name in hash
    ];

    let (result, count, _delta) = dedup_tool_results(msgs, 0, 3);
    // Different tool names produce different hashes, so NOT deduped
    assert_eq!(count, 0);
    if let Some(MessageContent::Text(t)) = &result[1].content {
        assert_eq!(t, content);
    }
    if let Some(MessageContent::Text(t)) = &result[2].content {
        assert_eq!(t, content);
    }
}

#[test]
fn dedup_small_content_empty_not_deduped() {
    // Empty content is skipped by the dedup logic
    let msgs = vec![
        make_assistant_with_tool_calls(vec![
            make_tool_call("tc1", "fn1", "{}"),
            make_tool_call("tc2", "fn1", "{}"),
        ]),
        ChatMessage::tool("tc1", ""),
        ChatMessage::tool("tc2", ""),
    ];

    let (result, count, _delta) = dedup_tool_results(msgs, 0, 3);
    assert_eq!(count, 0);
    // Both remain unchanged
    if let Some(MessageContent::Text(t)) = &result[1].content {
        assert_eq!(t, "");
    }
}

// ═══════════════════════════════════════════════════════════════
// truncate_tool_args
// ═══════════════════════════════════════════════════════════════

#[test]
fn truncate_long_json_string_values() {
    let long_val = "a".repeat(600);
    let args = format!(r#"{{"code":"{}","flag":true}}"#, long_val);
    assert!(args.len() > 500);

    let msgs = vec![
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "write_file", &args)]),
        ChatMessage::tool("tc1", "ok"),
    ];

    let (result, count, _delta) = truncate_tool_args(msgs, 0, 2);
    assert_eq!(count, 1);
    let new_args = &result[0].tool_calls.as_ref().unwrap()[0].function.arguments;
    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(new_args).expect("must be valid JSON");
    // The "code" field should be truncated
    let code = parsed["code"].as_str().unwrap();
    assert!(code.len() < 600);
    assert!(code.contains("...[truncated]"));
    // Boolean preserved
    assert_eq!(parsed["flag"], serde_json::Value::Bool(true));
}

#[test]
fn truncate_short_args_untouched() {
    let args = r#"{"path":"hello.rs"}"#;
    assert!(args.len() < 500);

    let msgs = vec![
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read", args)]),
        ChatMessage::tool("tc1", "content"),
    ];

    let (result, count, _delta) = truncate_tool_args(msgs, 0, 2);
    assert_eq!(count, 0);
    let kept = &result[0].tool_calls.as_ref().unwrap()[0].function.arguments;
    assert_eq!(kept, args);
}

#[test]
fn truncate_multibyte_utf8_no_panic() {
    // Create args with multi-byte characters exceeding threshold
    let emoji_val = "🎉".repeat(200); // each emoji is 4 bytes, total 800 bytes
    let args = format!(r#"{{"emoji":"{}"}}"#, emoji_val);
    assert!(args.len() > 500);

    let msgs = vec![
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "fun", &args)]),
        ChatMessage::tool("tc1", "ok"),
    ];

    let (result, count, _delta) = truncate_tool_args(msgs, 0, 2);
    assert_eq!(count, 1);
    let new_args = &result[0].tool_calls.as_ref().unwrap()[0].function.arguments;
    // Must be valid JSON (no panic on multi-byte boundary)
    let parsed: serde_json::Value =
        serde_json::from_str(new_args).expect("must be valid JSON after truncation");
    let emoji_field = parsed["emoji"].as_str().unwrap();
    assert!(emoji_field.contains("...[truncated]"));
}
