use std::collections::HashMap;

use cold_sdk::{ChatMessage, ContentPart, FunctionCall, MessageContent, Role, ToolCall};

use crate::util::content_text_length;

/// Strip images from messages before `tail_start`, replacing with `[image removed]`.
///
/// Takes ownership of the message vec to avoid unnecessary clones. Messages that
/// don't need modification pass through without cloning.
///
/// Returns `(new_messages, images_stripped_count, estimated_tokens_removed)`.
#[must_use]
pub fn strip_historical_images(
    messages: Vec<ChatMessage>,
    tail_start: usize,
) -> (Vec<ChatMessage>, usize, u32) {
    /// Estimated tokens per image (flat cost in `CharEstimator`).
    const IMAGE_TOKEN_COST: u32 = 1600;
    /// Estimated tokens for the replacement text "[image removed]".
    const REPLACEMENT_TOKENS: u32 = 4;

    let mut result = Vec::with_capacity(messages.len());
    let mut stripped = 0usize;
    let mut tokens_removed: u32 = 0;

    for (i, msg) in messages.into_iter().enumerate() {
        if i < tail_start {
            if let Some(MessageContent::Parts(ref parts)) = msg.content {
                let has_images = parts
                    .iter()
                    .any(|p| matches!(p, ContentPart::ImageUrl { .. }));
                if has_images {
                    let new_parts: Vec<ContentPart> = parts
                        .iter()
                        .map(|p| match p {
                            ContentPart::ImageUrl { .. } => {
                                stripped += 1;
                                tokens_removed +=
                                    IMAGE_TOKEN_COST.saturating_sub(REPLACEMENT_TOKENS);
                                ContentPart::Text {
                                    text: "[image removed]".to_string(),
                                }
                            }
                            other => other.clone(),
                        })
                        .collect();
                    let mut new_msg = msg;
                    new_msg.content = Some(MessageContent::Parts(new_parts));
                    result.push(new_msg);
                    continue;
                }
            }
        }
        result.push(msg);
    }

    (result, stripped, tokens_removed)
}

/// Prune long tool outputs in the middle zone by replacing with a short summary.
///
/// Takes ownership of the message vec to avoid unnecessary clones.
///
/// Returns `(new_messages, pruned_count, estimated_tokens_removed)`.
#[must_use]
pub fn prune_tool_outputs(
    messages: Vec<ChatMessage>,
    head_end: usize,
    tail_start: usize,
) -> (Vec<ChatMessage>, usize, u32) {
    const CONTENT_THRESHOLD: usize = 200;
    const CHARS_PER_TOKEN: usize = 3;

    // Build lookup: tool_call_id -> (function_name, args_preview)
    // We need to scan first, so borrow before consuming.
    let mut tool_call_lookup: HashMap<String, (String, String)> = HashMap::new();
    for msg in &messages {
        if msg.role == Role::Assistant {
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    let args_preview = if tc.function.arguments.len() > 80 {
                        let pos = crate::util::safe_truncate_pos(&tc.function.arguments, 80);
                        format!("{}...", &tc.function.arguments[..pos])
                    } else {
                        tc.function.arguments.clone()
                    };
                    tool_call_lookup
                        .insert(tc.id.clone(), (tc.function.name.clone(), args_preview));
                }
            }
        }
    }

    let mut result = Vec::with_capacity(messages.len());
    let mut pruned = 0usize;
    let mut tokens_removed: u32 = 0;

    for (i, msg) in messages.into_iter().enumerate() {
        if i >= head_end && i < tail_start && msg.role == Role::Tool {
            let content_len = content_text_length(msg.content.as_ref());
            if content_len > CONTENT_THRESHOLD {
                let (tool_name, args_preview) = msg
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| tool_call_lookup.get(id))
                    .map_or_else(
                        || ("unknown".to_string(), String::new()),
                        |(name, args)| (name.clone(), args.clone()),
                    );

                let replacement = format!("[{tool_name}] {args_preview} ({content_len} bytes)");

                #[allow(clippy::cast_possible_truncation)]
                let delta =
                    (content_len.saturating_sub(replacement.len()) / CHARS_PER_TOKEN) as u32;
                tokens_removed += delta;

                let mut new_msg = msg;
                new_msg.content = Some(MessageContent::Text(replacement));
                result.push(new_msg);
                pruned += 1;
                continue;
            }
        }
        result.push(msg);
    }

    (result, pruned, tokens_removed)
}

/// Deduplicate tool results in the middle zone with identical content.
///
/// Keeps the newest (last) occurrence, replaces older duplicates with a note.
/// Takes ownership of the message vec to avoid unnecessary clones.
///
/// Returns `(new_messages, deduped_count, estimated_tokens_removed)`.
#[must_use]
pub fn dedup_tool_results(
    messages: Vec<ChatMessage>,
    head_end: usize,
    tail_start: usize,
) -> (Vec<ChatMessage>, usize, u32) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // NOTE: DefaultHasher is not stable across Rust versions. This is acceptable
    // because the hash is ephemeral/within-process only and never persisted or shared.
    //
    // NOTE: The ~2^-64 collision risk from the 64-bit hash is acceptable for
    // deduplication of tool results in a single conversation.

    // Build lookup: tool_call_id -> tool_name (for identity-aware hashing)
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

    // First pass: find the last index for each unique (tool_name + content) hash in the middle zone
    let mut last_occurrence: HashMap<u64, usize> = HashMap::new();
    for (i, msg) in messages.iter().enumerate().take(tail_start).skip(head_end) {
        if msg.role == Role::Tool {
            let text = crate::util::extract_text(msg.content.as_ref());
            if text.is_empty() {
                continue;
            }
            let mut hasher = DefaultHasher::new();
            if let Some(tool_name) = msg
                .tool_call_id
                .as_deref()
                .and_then(|id| tool_name_lookup.get(id))
            {
                tool_name.hash(&mut hasher);
            }
            text.hash(&mut hasher);
            let hash = hasher.finish();
            last_occurrence.insert(hash, i);
        }
    }

    // Second pass: replace older duplicates
    #[allow(clippy::items_after_statements)]
    const DEDUP_REPLACEMENT: &str = "[Duplicate — same as more recent call]";
    #[allow(clippy::items_after_statements)]
    const CHARS_PER_TOKEN: usize = 3;

    let mut result = Vec::with_capacity(messages.len());
    let mut deduped = 0usize;
    let mut tokens_removed: u32 = 0;

    for (i, msg) in messages.into_iter().enumerate() {
        if i >= head_end && i < tail_start && msg.role == Role::Tool {
            let text = crate::util::extract_text(msg.content.as_ref());
            if !text.is_empty() {
                let mut hasher = DefaultHasher::new();
                if let Some(tool_name) = msg
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| tool_name_lookup.get(id))
                {
                    tool_name.hash(&mut hasher);
                }
                text.hash(&mut hasher);
                let hash = hasher.finish();

                if let Some(&last_idx) = last_occurrence.get(&hash) {
                    if last_idx != i {
                        #[allow(clippy::cast_possible_truncation)]
                        let delta = (text.len().saturating_sub(DEDUP_REPLACEMENT.len())
                            / CHARS_PER_TOKEN) as u32;
                        tokens_removed += delta;

                        let mut new_msg = msg;
                        new_msg.content = Some(MessageContent::Text(DEDUP_REPLACEMENT.to_string()));
                        result.push(new_msg);
                        deduped += 1;
                        continue;
                    }
                }
            }
        }
        result.push(msg);
    }

    (result, deduped, tokens_removed)
}

/// Truncate long tool call arguments in assistant messages within the middle zone.
///
/// If an argument string is > 500 chars, parse as JSON, truncate string values to 200 chars,
/// and re-serialize. Takes ownership of the message vec to avoid unnecessary clones.
///
/// Returns `(new_messages, truncated_count, estimated_tokens_removed)`.
///
#[must_use]
pub fn truncate_tool_args(
    messages: Vec<ChatMessage>,
    head_end: usize,
    tail_start: usize,
) -> (Vec<ChatMessage>, usize, u32) {
    const ARGS_THRESHOLD: usize = 500;
    const STRING_MAX: usize = 200;
    const CHARS_PER_TOKEN: usize = 3;

    let mut result = Vec::with_capacity(messages.len());
    let mut truncated = 0usize;
    let mut tokens_removed: u32 = 0;

    for (i, msg) in messages.into_iter().enumerate() {
        if i >= head_end && i < tail_start && msg.role == Role::Assistant {
            let Some(tool_calls) = msg.tool_calls.as_ref() else {
                result.push(msg);
                continue;
            };

            let mut any_changed = false;
            let mut msg_delta: usize = 0;
            let new_tool_calls: Vec<ToolCall> = tool_calls
                .iter()
                .map(|tc| {
                    if tc.function.arguments.len() > ARGS_THRESHOLD {
                        if let Ok(mut val) =
                            serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                        {
                            truncate_json_strings(&mut val, STRING_MAX);
                            if let Ok(new_args) = serde_json::to_string(&val) {
                                let old_len = tc.function.arguments.len();
                                msg_delta += old_len.saturating_sub(new_args.len());
                                any_changed = true;
                                return ToolCall {
                                    id: tc.id.clone(),
                                    call_type: tc.call_type.clone(),
                                    function: FunctionCall {
                                        name: tc.function.name.clone(),
                                        arguments: new_args,
                                    },
                                };
                            }
                        }
                    }
                    tc.clone()
                })
                .collect();

            if any_changed {
                #[allow(clippy::cast_possible_truncation)]
                let delta = (msg_delta / CHARS_PER_TOKEN) as u32;
                tokens_removed += delta;
                let mut new_msg = msg;
                new_msg.tool_calls = Some(new_tool_calls);
                result.push(new_msg);
                truncated += 1;
                continue;
            }
        }
        result.push(msg);
    }

    (result, truncated, tokens_removed)
}

fn truncate_json_strings(value: &mut serde_json::Value, max_len: usize) {
    match value {
        serde_json::Value::String(s) if s.len() > max_len => {
            let pos = crate::util::safe_truncate_pos(s, max_len);
            s.truncate(pos);
            s.push_str("...[truncated]");
        }
        serde_json::Value::Array(arr) => {
            for v in arr {
                truncate_json_strings(v, max_len);
            }
        }
        serde_json::Value::Object(map) => {
            for v in map.values_mut() {
                truncate_json_strings(v, max_len);
            }
        }
        _ => {}
    }
}
