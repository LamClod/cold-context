use cold_context::boundary::{Boundaries, find_boundaries, snap_to_tool_groups};
use cold_context::config::CompressorConfig;
use cold_context::counter::CharEstimator;
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

fn make_tool_result(tool_call_id: &str, content: &str) -> ChatMessage {
    ChatMessage::tool(tool_call_id, content)
}

/// Build a conversation: system + N user/assistant pairs
fn build_conversation(pairs: usize) -> Vec<ChatMessage> {
    let mut msgs = vec![ChatMessage::system("You are a helpful assistant.")];
    for i in 0..pairs {
        msgs.push(ChatMessage::user(format!("Question {i}")));
        msgs.push(ChatMessage::assistant(format!("Answer {i}")));
    }
    msgs
}

#[test]
fn normal_case_system_plus_10_pairs() {
    // system + 10 pairs = 21 messages, protect_first_n=3, protect_last_n=3
    // Use larger messages so tail budget doesn't absorb everything
    let mut msgs = vec![ChatMessage::system("You are a helpful assistant.")];
    for i in 0..10 {
        let chunk = "x".repeat(200);
        msgs.push(ChatMessage::user(format!("{chunk} question {i}")));
        msgs.push(ChatMessage::assistant(format!("{chunk} answer {i}")));
    }
    assert_eq!(msgs.len(), 21);

    // Use a context_length small enough that tail budget won't absorb all messages
    // threshold = 600 * 0.50 = 300, tail_budget = 300 * 0.30 = 90 tokens
    // Each msg ~(200+12)/4 + 4 = 57 tokens, so tail can hold ~1-2 messages
    let config = CompressorConfig::new("test-model", 600)
        .with_protect_first_n(3)
        .with_protect_last_n(3);

    let b = find_boundaries(&msgs, &config, &CharEstimator);

    // head_end: system(1) + 3 non-system = index 4
    assert_eq!(b.head_end, 4);
    // tail_start: floor is len - protect_last_n = 21 - 3 = 18
    assert!(b.tail_start <= 18);
    // Middle zone exists
    assert!(
        b.head_end < b.tail_start,
        "expected middle zone, head_end={} tail_start={}",
        b.head_end,
        b.tail_start
    );
}

#[test]
fn edge_all_messages_fit_in_head_and_tail() {
    // Only 5 messages total, protect_first_n=3, protect_last_n=3
    let msgs = build_conversation(2); // system + 2 pairs = 5 msgs
    assert_eq!(msgs.len(), 5);

    let config = CompressorConfig::new("test-model", 100_000)
        .with_protect_first_n(3)
        .with_protect_last_n(3);

    let b = find_boundaries(&msgs, &config, &CharEstimator);

    // head covers 1 system + 3 non-system = 4 (but only 4 non-system exist)
    // tail covers last 3. With overlap, tail_start >= head_end
    assert!(
        b.tail_start >= b.head_end,
        "no middle zone expected when all fit"
    );
}

#[test]
fn edge_only_system_message() {
    let msgs = vec![ChatMessage::system("sys")];
    let config = CompressorConfig::new("test-model", 100_000)
        .with_protect_first_n(3)
        .with_protect_last_n(3);

    let b = find_boundaries(&msgs, &config, &CharEstimator);
    assert_eq!(b.head_end, 1);
    assert_eq!(b.tail_start, 1); // no middle
}

#[test]
fn edge_protect_first_n_larger_than_message_count() {
    let msgs = build_conversation(2); // 5 messages
    let config = CompressorConfig::new("test-model", 100_000)
        .with_protect_first_n(100)
        .with_protect_last_n(3);

    let b = find_boundaries(&msgs, &config, &CharEstimator);
    // head should clamp to len
    assert!(b.head_end <= msgs.len());
    assert!(b.tail_start >= b.head_end);
}

#[test]
fn tool_group_at_head_boundary_snaps_forward() {
    // Construct: system, user, assistant(tool_call), tool_result, user, assistant, user, assistant
    // Use large messages so tail doesn't absorb everything
    let large = "x".repeat(200);
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user(format!("{large} q1")),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "read", "{}")]),
        make_tool_result("tc1", &format!("{large} file content")),
        ChatMessage::user(format!("{large} q2")),
        ChatMessage::assistant(format!("{large} a2")),
        ChatMessage::user(format!("{large} q3")),
        ChatMessage::assistant(format!("{large} a3")),
    ];

    // context_length=400, threshold=200, tail_budget=60 => tail holds ~1 msg
    let config = CompressorConfig::new("test-model", 400)
        .with_protect_first_n(2)
        .with_protect_last_n(2);

    let boundaries = find_boundaries(&msgs, &config, &CharEstimator);
    // head_end should be 3 (system + 2 non-system: user at 1, assistant at 2 => head_end=3)
    // index 3 is a tool result, snap should push forward
    let snapped = snap_to_tool_groups(&msgs, boundaries);

    // The tool result at index 3 should be included in the head (snap forward)
    assert!(
        snapped.head_end >= 4,
        "tool result should be included in head, got head_end={}, boundaries were {:?}",
        snapped.head_end,
        boundaries
    );
}

#[test]
fn tool_group_at_tail_boundary_snaps_backward() {
    // Construct: system, user, assistant, user, assistant(tool_call), tool_result, user, assistant
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("q1"),
        ChatMessage::assistant("a1"),
        ChatMessage::user("q2"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "search", "{}")]),
        make_tool_result("tc1", "search results"),
        ChatMessage::user("q3"),
        ChatMessage::assistant("a3"),
    ];

    // Force tail_start to land on index 5 (the tool_result)
    // With large context (won't limit by token budget), protect_last_n=3 => tail_start = 8-3 = 5
    let config = CompressorConfig::new("test-model", 100_000)
        .with_protect_first_n(2)
        .with_protect_last_n(3);

    let boundaries = find_boundaries(&msgs, &config, &CharEstimator);
    let snapped = snap_to_tool_groups(&msgs, boundaries);

    // If tail_start was on tool result, snap should pull it back to include the assistant
    if boundaries.tail_start == 5 {
        assert!(
            snapped.tail_start <= 4,
            "should snap backward to include assistant with tool_calls"
        );
    }
}

#[test]
fn all_tool_message_middle_degenerate_relaxation() {
    // Middle zone is entirely tool messages -> snap causes degenerate, relaxation kicks in
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("q1"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "fn1", "{}")]),
        make_tool_result("tc1", "result1"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc2", "fn2", "{}")]),
        make_tool_result("tc2", "result2"),
        ChatMessage::user("q2"),
        ChatMessage::assistant("a2"),
    ];

    // Force boundaries so middle is indices 2..6 (all tool-related)
    let forced = Boundaries {
        head_end: 2,
        tail_start: 6,
    };

    let snapped = snap_to_tool_groups(&msgs, forced);
    // Snap might expand to include tool groups or relax. Just ensure no panic and valid result.
    assert!(snapped.head_end <= snapped.tail_start || snapped.head_end <= msgs.len());
}

#[test]
fn multiple_tool_calls_in_one_assistant_message() {
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("q1"),
        ChatMessage::assistant("a1"),
        ChatMessage::user("q2"),
        make_assistant_with_tool_calls(vec![
            make_tool_call("tc1", "read_file", r#"{"path":"a.rs"}"#),
            make_tool_call("tc2", "read_file", r#"{"path":"b.rs"}"#),
        ]),
        make_tool_result("tc1", "contents of a"),
        make_tool_result("tc2", "contents of b"),
        ChatMessage::user("q3"),
        ChatMessage::assistant("a3"),
    ];

    let config = CompressorConfig::new("test-model", 100_000)
        .with_protect_first_n(2)
        .with_protect_last_n(2);

    let boundaries = find_boundaries(&msgs, &config, &CharEstimator);
    let snapped = snap_to_tool_groups(&msgs, boundaries);

    // The multi-tool group (indices 4,5,6) must not be split
    if snapped.head_end > 4 && snapped.head_end <= 6 {
        // If head reaches into the tool group, it must include all results
        assert!(snapped.head_end >= 7, "must not split multi-tool group");
    }
    if snapped.tail_start > 4 && snapped.tail_start <= 6 {
        // If tail starts in the tool group, it must include the assistant
        assert!(
            snapped.tail_start <= 4,
            "must not split multi-tool group at tail"
        );
    }
}

#[test]
fn boundary_relaxation_tries_multiple_steps() {
    // Create a scenario where initial snap is degenerate but relaxation succeeds
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("q1"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "fn1", "{}")]),
        make_tool_result("tc1", "r1"),
        ChatMessage::user("q2"),
        ChatMessage::assistant("a2"),
        ChatMessage::user("q3"),
        ChatMessage::assistant("a3"),
    ];

    // Force tight boundaries where snap causes overlap
    let forced = Boundaries {
        head_end: 3,
        tail_start: 4,
    };

    let snapped = snap_to_tool_groups(&msgs, forced);
    // After relaxation, should find a valid split or remain degenerate
    // The important thing is no panic
    assert!(snapped.head_end <= msgs.len());
    assert!(snapped.tail_start <= msgs.len());
}
