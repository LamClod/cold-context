use cold_context::util::{content_text_length, extract_text, safe_truncate_pos};
use cold_sdk::{ContentPart, ImageUrl, InputAudio, MessageContent};

// ═══════════════════════════════════════════════════════════════
// content_text_length
// ═══════════════════════════════════════════════════════════════

#[test]
fn content_text_length_text_variant() {
    let content = MessageContent::Text("hello world".to_string());
    assert_eq!(content_text_length(Some(&content)), 11);
}

#[test]
fn content_text_length_parts_with_image() {
    let content = MessageContent::Parts(vec![
        ContentPart::Text {
            text: "look:".to_string(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://img.png".to_string(),
                detail: None,
            },
        },
    ]);
    // "look:" = 5 bytes + image = 4800
    assert_eq!(content_text_length(Some(&content)), 5 + 4800);
}

#[test]
fn content_text_length_parts_with_audio() {
    let content = MessageContent::Parts(vec![ContentPart::InputAudio {
        input_audio: InputAudio {
            data: "base64".to_string(),
            format: "wav".to_string(),
        },
    }]);
    // audio = 4800
    assert_eq!(content_text_length(Some(&content)), 4800);
}

#[test]
fn content_text_length_none() {
    assert_eq!(content_text_length(None), 0);
}

// ═══════════════════════════════════════════════════════════════
// extract_text
// ═══════════════════════════════════════════════════════════════

#[test]
fn extract_text_from_text_content() {
    let content = MessageContent::Text("hello".to_string());
    assert_eq!(extract_text(Some(&content)), "hello");
}

#[test]
fn extract_text_from_parts_with_image() {
    let content = MessageContent::Parts(vec![
        ContentPart::Text {
            text: "before".to_string(),
        },
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: "https://img.png".to_string(),
                detail: None,
            },
        },
        ContentPart::Text {
            text: "after".to_string(),
        },
    ]);
    assert_eq!(extract_text(Some(&content)), "before\n[image]\nafter");
}

#[test]
fn extract_text_from_parts_with_audio() {
    let content = MessageContent::Parts(vec![ContentPart::InputAudio {
        input_audio: InputAudio {
            data: "d".to_string(),
            format: "mp3".to_string(),
        },
    }]);
    assert_eq!(extract_text(Some(&content)), "[audio]");
}

#[test]
fn extract_text_none() {
    assert_eq!(extract_text(None), "");
}

// ═══════════════════════════════════════════════════════════════
// safe_truncate_pos
// ═══════════════════════════════════════════════════════════════

#[test]
fn safe_truncate_ascii() {
    let s = "hello world";
    assert_eq!(safe_truncate_pos(s, 5), 5);
    assert_eq!(&s[..safe_truncate_pos(s, 5)], "hello");
}

#[test]
fn safe_truncate_cjk() {
    let s = "你好世界"; // each char is 3 bytes: 0,3,6,9 are boundaries
    // Trying to cut at byte 4 (middle of second char) should snap back to 3
    assert_eq!(safe_truncate_pos(s, 4), 3);
    assert_eq!(&s[..safe_truncate_pos(s, 4)], "你");
    // Cutting at byte 6 is a boundary
    assert_eq!(safe_truncate_pos(s, 6), 6);
    assert_eq!(&s[..safe_truncate_pos(s, 6)], "你好");
}

#[test]
fn safe_truncate_emoji() {
    let s = "Hi🎉Bye"; // 'H'=1, 'i'=1, '🎉'=4, 'B'=1, 'y'=1, 'e'=1 => total 9 bytes
    // Cutting at byte 3 (middle of emoji) should snap back to 2
    assert_eq!(safe_truncate_pos(s, 3), 2);
    assert_eq!(&s[..safe_truncate_pos(s, 3)], "Hi");
    // Cutting at byte 6 (after emoji) is a boundary
    assert_eq!(safe_truncate_pos(s, 6), 6);
    assert_eq!(&s[..safe_truncate_pos(s, 6)], "Hi🎉");
}

#[test]
fn safe_truncate_at_zero() {
    let s = "hello";
    assert_eq!(safe_truncate_pos(s, 0), 0);
}

#[test]
fn safe_truncate_beyond_length() {
    let s = "hi";
    assert_eq!(safe_truncate_pos(s, 100), 2);
}

#[test]
fn safe_truncate_empty_string() {
    assert_eq!(safe_truncate_pos("", 5), 0);
}
