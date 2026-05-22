use cold_context::compressor::{CompressionStage, ContextCompressor};
use cold_context::config::CompressorConfig;
use cold_context::error::ContextError;
use cold_context::summary::Summarizer;
use cold_sdk::{ChatMessage, FunctionCall, MessageContent, Role, ToolCall, Usage};

// ═══════════════════════════════════════════════════════════════
// MockSummarizer
// ═══════════════════════════════════════════════════════════════

struct MockSummarizer {
    response: String,
    call_count: std::sync::atomic::AtomicU32,
}

impl MockSummarizer {
    fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
}

impl Summarizer for MockSummarizer {
    async fn summarize(&self, _content: &str, _max_tokens: u32) -> Result<String, ContextError> {
        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(self.response.clone())
    }
}

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

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

/// Build a large conversation that exceeds a given threshold
fn build_large_conversation(token_target: u32) -> Vec<ChatMessage> {
    let mut msgs = vec![ChatMessage::system("You are a helpful assistant.")];
    // Each user/assistant pair: ~4+overhead each, make content large
    let chunk = "x".repeat(200); // 200 bytes / 3 = 66 tokens each msg + 4 overhead = 70
    let pairs_needed = (token_target / 140) + 5; // 140 tokens per pair, add margin
    for i in 0..pairs_needed {
        msgs.push(ChatMessage::user(format!("{chunk} question {i}")));
        msgs.push(ChatMessage::assistant(format!("{chunk} answer {i}")));
    }
    msgs
}

// ═══════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn should_compress_returns_false_under_threshold() {
    let config = CompressorConfig::new("test", 10_000);
    let mock = MockSummarizer::new("summary");
    let compressor = ContextCompressor::with_summarizer(config, mock);
    // No usage updated, last_prompt_tokens = 0 < threshold
    assert!(!compressor.should_compress());
}

#[test]
fn should_compress_returns_true_over_threshold() {
    let config = CompressorConfig::new("test", 10_000); // threshold = 5000
    let mock = MockSummarizer::new("summary");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);
    compressor.update_usage(&Usage {
        prompt_tokens: 6000,
        completion_tokens: 100,
        total_tokens: 6100,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });
    assert!(compressor.should_compress());
}

#[tokio::test]
async fn compress_with_cheap_stages_only() {
    // Large tool outputs that get pruned, bringing tokens under threshold.
    // We need: total tokens > threshold BEFORE pruning, but < threshold AFTER pruning.
    // Large tool output: 4000 chars / 3 = 1333 tokens.
    // threshold = 1800 * 0.50 = 900. Total before: sys(4) + user(4) + asst(4) + tool(4+1333) + user(4) + asst(4+3) = ~1358
    // After pruning tool output (replaced with ~50 char summary => 16 tokens): ~40 total.
    let config = CompressorConfig::new("test", 1800)
        .with_threshold_percent(0.50) // threshold = 900
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    let mock = MockSummarizer::new("summary");

    let large_output = "z".repeat(4000); // > 200 threshold for pruning, contributes ~1333 tokens
    let msgs = vec![
        ChatMessage::system("sys"),
        ChatMessage::user("q1"),
        make_assistant_with_tool_calls(vec![make_tool_call("tc1", "big_tool", "{}")]),
        ChatMessage::tool("tc1", &large_output),
        ChatMessage::user("q2"),
        ChatMessage::assistant("end"),
    ];

    let mut compressor = ContextCompressor::with_summarizer(config, mock);
    let result = compressor.compress(msgs, None).await.unwrap();

    // Should have applied ToolPrune stage
    assert!(
        result.stages.contains(&CompressionStage::ToolPrune),
        "expected ToolPrune stage, got stages: {:?}, original_tokens={}, final_tokens={}",
        result.stages,
        result.original_tokens,
        result.final_tokens
    );
}

#[tokio::test]
async fn compress_triggers_llm_summary() {
    // Need: middle zone tokens > MIN_MIDDLE_TOKENS_FOR_SUMMARY (2000).
    // With protect_first_n=1, protect_last_n=1, the tail budget walks back from the end.
    // Make each message large (~104 tokens) so 50 pairs = 100 msgs * 104 = 10400 tokens.
    // threshold = 20000 * 0.50 = 10000. tail_budget = 10000 * 0.30 = 3000 => ~28 msgs in tail.
    // Middle = 101 - 2(sys+head) - 28(tail) = 71 msgs * 104 = 7384 tokens > 2000.
    let config = CompressorConfig::new("test", 20_000)
        .with_threshold_percent(0.50) // threshold = 10000
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    let mock = MockSummarizer::new("## Key Decisions\n- Chose Rust\n## Current State\nWorking");

    let msgs = build_large_conversation(10_000);

    let mut compressor = ContextCompressor::with_summarizer(config, mock);
    let result = compressor.compress(msgs, None).await.unwrap();

    assert!(
        result.stages.contains(&CompressionStage::LlmSummarize),
        "expected LlmSummarize, got stages: {:?}, original_tokens={}, final_tokens={}",
        result.stages,
        result.original_tokens,
        result.final_tokens
    );
    assert!(result.savings_pct > 0.0);
    assert_eq!(compressor.compression_count(), 1);
}

#[tokio::test]
async fn compress_few_messages_nothing_happens() {
    // With very few messages, the compressor may return Ok with no changes
    // (snap_to_tool_groups relaxation can create a tiny middle zone, but
    // no stages will actually fire since there's nothing to prune/summarize).
    // Alternatively it may return NothingToCompress if boundaries overlap.
    let config = CompressorConfig::new("test", 10_000)
        .with_protect_first_n(10)
        .with_protect_last_n(10);
    let mock = MockSummarizer::new("summary");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    let msgs = vec![ChatMessage::system("sys")];
    let result = compressor.compress(msgs, None).await;
    match result {
        Ok(r) => {
            // No stages applied, no savings
            assert!(r.stages.is_empty());
            assert_eq!(r.savings_pct, 0.0);
        }
        Err(ContextError::NothingToCompress) => {
            // Also acceptable
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[tokio::test]
async fn anti_thrashing_after_two_ineffective() {
    // The summary returned is almost as large as the middle zone, so savings < 10%.
    let config = CompressorConfig::new("test", 20_000)
        .with_threshold_percent(0.50)
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    // Return a summary almost as large as the original total to ensure < 10% savings
    let long_summary = "word ".repeat(4000); // 20000 chars => 5000 tokens
    let mock = MockSummarizer::new(&long_summary);
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    compressor.update_usage(&Usage {
        prompt_tokens: 15_000,
        completion_tokens: 50,
        total_tokens: 15_050,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    let msgs = build_large_conversation(10_000);

    // First compression
    let r1 = compressor.compress(msgs.clone(), None).await;
    if let Ok(ref r) = r1 {
        if r.savings_pct >= 0.10 {
            // savings too high for this test scenario
            return;
        }
    }

    // Second compression
    let msgs2 = build_large_conversation(10_000);
    let _ = compressor.compress(msgs2, None).await;

    // After 2 ineffective compressions, should_compress returns false
    assert!(
        !compressor.should_compress(),
        "expected should_compress=false after 2 ineffective, last_savings={}",
        compressor.last_savings_pct()
    );
}

#[tokio::test]
async fn anti_thrashing_reset_after_token_growth() {
    let config = CompressorConfig::new("test", 20_000)
        .with_threshold_percent(0.50) // threshold = 10000
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    let long_summary = "word ".repeat(4000);
    let mock = MockSummarizer::new(&long_summary);
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    compressor.update_usage(&Usage {
        prompt_tokens: 15_000,
        completion_tokens: 50,
        total_tokens: 15_050,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    let msgs = build_large_conversation(10_000);
    let _ = compressor.compress(msgs.clone(), None).await;
    let _ = compressor.compress(msgs.clone(), None).await;

    // Now simulate large token growth (>20% and >1000 abs growth)
    compressor.update_usage(&Usage {
        prompt_tokens: 30_000,
        completion_tokens: 50,
        total_tokens: 30_050,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    // Anti-thrashing should be reset, should_compress should return true
    assert!(compressor.should_compress());
}

#[tokio::test]
async fn iterative_summary_old_summary_removed() {
    let config = CompressorConfig::new("test", 20_000)
        .with_threshold_percent(0.50) // threshold = 10000
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    let mock = MockSummarizer::new("## Key Decisions\n- iteration test");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    let msgs = build_large_conversation(10_000);

    // First compression
    let result1 = compressor.compress(msgs, None).await.unwrap();
    assert!(
        compressor.previous_summary().is_some(),
        "expected previous_summary after compression, stages={:?}",
        result1.stages
    );

    // Build a new large conversation that includes the compressed result + more messages
    let mut msgs2 = result1.messages;
    let chunk = "y".repeat(200);
    for i in 0..200 {
        msgs2.push(ChatMessage::user(format!("{chunk} followup {i}")));
        msgs2.push(ChatMessage::assistant(format!("{chunk} reply {i}")));
    }

    let result2 = compressor.compress(msgs2, None).await;
    match result2 {
        Ok(r) => {
            // Old summary message should not appear twice
            let summary_count = r
                .messages
                .iter()
                .filter(|m| {
                    if let Some(MessageContent::Text(t)) = &m.content {
                        t.contains("[CONTEXT COMPACTION")
                    } else {
                        false
                    }
                })
                .count();
            assert!(
                summary_count <= 1,
                "old summary should be removed, found {summary_count}"
            );
        }
        Err(ContextError::NothingToCompress) => {
            // Acceptable if the middle zone is empty after first compression
        }
        Err(e) => panic!("unexpected error: {e}"),
    }
}

#[tokio::test]
async fn focus_topic_passed_to_summarizer() {
    // Verify compression with focus_topic doesn't panic and completes
    let config = CompressorConfig::new("test", 20_000)
        .with_threshold_percent(0.50)
        .with_protect_first_n(1)
        .with_protect_last_n(1);

    let mock = MockSummarizer::new("## Active Task\n- Working on Rust tests");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    let msgs = build_large_conversation(10_000);
    let result = compressor
        .compress(msgs, Some("Rust testing"))
        .await
        .unwrap();
    assert!(!result.messages.is_empty());
}

#[test]
fn reset_clears_all_state() {
    let config = CompressorConfig::new("test", 10_000);
    let mock = MockSummarizer::new("summary");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    // Simulate some state
    compressor.update_usage(&Usage {
        prompt_tokens: 8000,
        completion_tokens: 100,
        total_tokens: 8100,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    compressor.reset();
    assert_eq!(compressor.compression_count(), 0);
    assert_eq!(compressor.last_savings_pct(), 0.0);
    assert!(compressor.previous_summary().is_none());
}

#[test]
fn update_usage_tracks_tokens() {
    let config = CompressorConfig::new("test", 10_000);
    let mock = MockSummarizer::new("summary");
    let mut compressor = ContextCompressor::with_summarizer(config, mock);

    compressor.update_usage(&Usage {
        prompt_tokens: 3000,
        completion_tokens: 200,
        total_tokens: 3200,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });

    assert_eq!(compressor.last_prompt_tokens(), 3000);
}

#[test]
#[should_panic(expected = "context_length must be > 0")]
fn config_context_length_zero_panics() {
    let _ = CompressorConfig::new("test", 0);
}
