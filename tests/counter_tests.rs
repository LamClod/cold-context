use cold_context::counter::{CharEstimator, TokenCounter};
use cold_sdk::{ChatMessage, ContentPart, FunctionCall, ImageUrl, InputAudio, ToolCall};

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

#[test]
fn ascii_text_message_token_count() {
    let estimator = CharEstimator;
    // 20 chars / 3 = 6 tokens + 4 overhead = 10
    let msg = ChatMessage::user("Hello, world! 123456");
    assert_eq!(estimator.count_message(&msg), 4 + 20 / 3);
}

#[test]
fn cjk_text_message_token_count() {
    let estimator = CharEstimator;
    // CJK chars are 3 bytes each in UTF-8, but CharEstimator counts chars by len() which is byte length
    // "你好世界" = 12 bytes, 12/3 = 4 tokens + 4 overhead = 8
    let msg = ChatMessage::user("你好世界");
    let byte_len = "你好世界".len(); // 12
    assert_eq!(estimator.count_message(&msg), 4 + (byte_len as u32 / 3));
}

#[test]
fn mixed_ascii_cjk_token_count() {
    let estimator = CharEstimator;
    // "Hello你好" = 5 + 6 = 11 bytes, 11/3 = 3 tokens + 4 overhead = 7
    let msg = ChatMessage::user("Hello你好");
    let byte_len = "Hello你好".len(); // 11
    assert_eq!(estimator.count_message(&msg), 4 + (byte_len as u32 / 3));
}

#[test]
fn image_part_counts_as_1600_tokens() {
    let estimator = CharEstimator;
    let msg = ChatMessage::user_multimodal(vec![ContentPart::ImageUrl {
        image_url: ImageUrl {
            url: "https://example.com/img.png".to_string(),
            detail: None,
        },
    }]);
    // 1600 image + 4 overhead = 1604
    assert_eq!(estimator.count_message(&msg), 4 + 1600);
}

#[test]
fn audio_part_counts_as_1600_tokens() {
    let estimator = CharEstimator;
    let msg = ChatMessage::user_multimodal(vec![ContentPart::InputAudio {
        input_audio: InputAudio {
            data: "base64data".to_string(),
            format: "wav".to_string(),
        },
    }]);
    // 1600 audio + 4 overhead = 1604
    assert_eq!(estimator.count_message(&msg), 4 + 1600);
}

#[test]
fn tool_calls_add_arguments_length_div_3() {
    let estimator = CharEstimator;
    // arguments = 40 chars => 13 tokens; name "read_file" = 9 chars => 3 tokens; id "call_1" = 6 => 2 tokens
    let args = "a".repeat(40);
    let msg = ChatMessage {
        role: cold_sdk::Role::Assistant,
        content: None,
        name: None,
        tool_calls: Some(vec![make_tool_call("call_1", "read_file", &args)]),
        tool_call_id: None,
        refusal: None,
    };
    // overhead(4) + args(40/3=13) + name(9/3=3) + id(6/3=2) = 22
    assert_eq!(estimator.count_message(&msg), 4 + 40 / 3 + 9 / 3 + 6 / 3);
}

#[test]
fn empty_message_counts_overhead_only() {
    let estimator = CharEstimator;
    let msg = ChatMessage {
        role: cold_sdk::Role::Assistant,
        content: None,
        name: None,
        tool_calls: None,
        tool_call_id: None,
        refusal: None,
    };
    assert_eq!(estimator.count_message(&msg), 4);
}

#[test]
fn count_messages_sums_correctly() {
    let estimator = CharEstimator;
    let messages = vec![
        ChatMessage::user("Hello, world! 123456"), // 4 + 20/3=6 = 10
        ChatMessage::assistant("Hi there!"),       // 4 + 9/3=3 = 7
    ];
    let sum = estimator.count_message(&messages[0]) + estimator.count_message(&messages[1]);
    assert_eq!(estimator.count_messages(&messages), sum);
}

#[test]
fn multi_part_message_text_image_text() {
    let estimator = CharEstimator;
    let msg = ChatMessage::user_multimodal(vec![
        ContentPart::Text {
            text: "Look at this:".to_string(), // 13 bytes / 3 = 4
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://example.com/pic.png".to_string(),
                detail: None,
            },
        }, // 1600
        ContentPart::Text {
            text: "What do you see?".to_string(), // 16 bytes / 3 = 5
        },
    ]);
    // 4 overhead + 13/3 + 1600 + 16/3 = 4 + 4 + 1600 + 5 = 1613
    assert_eq!(estimator.count_message(&msg), 4 + 13 / 3 + 1600 + 16 / 3);
}
