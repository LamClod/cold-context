use cold_sdk::{ChatMessage, Role};

use crate::config::CompressorConfig;
use crate::counter::TokenCounter;

/// Indices marking the protected head and tail regions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Boundaries {
    /// Exclusive upper bound: `messages[..head_end]` are protected head.
    pub head_end: usize,
    /// Inclusive lower bound: `messages[tail_start..]` are protected tail.
    pub tail_start: usize,
}

/// Find the head/tail boundary indices.
///
/// - `head_end`: skip system messages, then protect `config.protect_first_n` non-system messages.
/// - `tail_start`: walk backward from end, accumulate token estimates, stop when
///   `tail_token_budget` is exceeded. Floor at `config.protect_last_n`.
pub fn find_boundaries(
    messages: &[ChatMessage],
    config: &CompressorConfig,
    counter: &dyn TokenCounter,
) -> Boundaries {
    let len = messages.len();

    // -- head_end --
    let mut head_end = 0;
    let mut non_system_seen = 0usize;
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == Role::System {
            head_end = i + 1;
            continue;
        }
        non_system_seen += 1;
        if non_system_seen >= config.protect_first_n {
            head_end = i + 1;
            break;
        }
        head_end = i + 1;
    }
    // Clamp
    head_end = head_end.min(len);

    // -- tail_start --
    #[allow(clippy::items_after_statements)]
    const TAIL_RATIO: f64 = 0.30;
    let threshold_tokens = config.threshold_tokens();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let tail_token_budget = (f64::from(threshold_tokens) * TAIL_RATIO) as u32;

    let mut tail_start = len;
    let mut accumulated: u32 = 0;

    // Walk backward
    for i in (0..len).rev() {
        let cost = counter.count_message(&messages[i]);
        if accumulated + cost > tail_token_budget && len - tail_start >= config.protect_last_n {
            break;
        }
        accumulated += cost;
        tail_start = i;

        // Ensure we protect at least protect_last_n messages
        if len - tail_start >= config.protect_last_n && accumulated >= tail_token_budget {
            break;
        }
    }

    // Floor: at minimum protect_last_n messages from the end
    let min_tail_start = len.saturating_sub(config.protect_last_n);
    if tail_start > min_tail_start {
        tail_start = min_tail_start;
    }

    // Ensure no overlap: tail_start must be >= head_end
    if tail_start < head_end {
        tail_start = head_end;
    }

    Boundaries {
        head_end,
        tail_start,
    }
}

/// Adjust boundaries so they don't cut inside a tool group.
///
/// A tool group = one assistant message with `tool_calls` + all following `Tool` role messages
/// until the next non-tool message.
///
/// W-6: After snapping, if the middle is degenerate (`head_end >= tail_start`), try relaxing
/// up to 3 steps in each direction. First shrinks the head, then expands the tail,
/// then tries combinations. Only gives up after all relaxation attempts fail.
///
/// The maximum relaxation in each direction is 3 steps.
#[must_use]
pub fn snap_to_tool_groups(messages: &[ChatMessage], boundaries: Boundaries) -> Boundaries {
    const MAX_RELAX: usize = 3;

    let result = snap_inner(messages, boundaries);
    if result.head_end < result.tail_start {
        return result;
    }

    // Try relaxing head only (up to MAX_RELAX steps).
    for h in 1..=MAX_RELAX {
        if boundaries.head_end >= h {
            let relaxed = Boundaries {
                head_end: boundaries.head_end - h,
                tail_start: boundaries.tail_start,
            };
            let result = snap_inner(messages, relaxed);
            if result.head_end < result.tail_start {
                return result;
            }
        }
    }

    // Try relaxing tail only (up to MAX_RELAX steps).
    for t in 1..=MAX_RELAX {
        if boundaries.tail_start + t <= messages.len() {
            let relaxed = Boundaries {
                head_end: boundaries.head_end,
                tail_start: boundaries.tail_start + t,
            };
            let result = snap_inner(messages, relaxed);
            if result.head_end < result.tail_start {
                return result;
            }
        }
    }

    // Try combined relaxation (head and tail together).
    for h in 1..=MAX_RELAX {
        for t in 1..=MAX_RELAX {
            let head = boundaries.head_end.saturating_sub(h);
            let tail = (boundaries.tail_start + t).min(messages.len());
            let relaxed = Boundaries {
                head_end: head,
                tail_start: tail,
            };
            let result = snap_inner(messages, relaxed);
            if result.head_end < result.tail_start {
                return result;
            }
        }
    }

    // All relaxations failed — return as-is (degenerate).
    snap_inner(messages, boundaries)
}

/// Core snapping logic, factored out so it can be retried with relaxed boundaries.
fn snap_inner(messages: &[ChatMessage], boundaries: Boundaries) -> Boundaries {
    let Boundaries {
        mut head_end,
        mut tail_start,
    } = boundaries;
    let len = messages.len();

    // -- Fix head_end: push forward to include orphaned tool results --
    while head_end < len && head_end < tail_start && messages[head_end].role == Role::Tool {
        head_end += 1;
    }

    // -- Fix tail_start: pull backward to include orphaned tool_uses --
    if tail_start > 0 && tail_start < len && messages[tail_start].role == Role::Tool {
        let mut scan = tail_start;
        while scan > head_end {
            scan -= 1;
            if messages[scan].role == Role::Assistant && messages[scan].tool_calls.is_some() {
                tail_start = scan;
                break;
            }
            if messages[scan].role != Role::Tool {
                break;
            }
        }
    }

    // Also check if messages[tail_start - 1] is an assistant with tool_calls
    // whose tool_results are in the tail zone.
    if tail_start > head_end {
        if let Some(prev) = messages.get(tail_start.wrapping_sub(1)) {
            if prev.role == Role::Assistant && prev.tool_calls.is_some() {
                if let Some(ref tool_calls) = prev.tool_calls {
                    let has_result_in_tail = tool_calls.iter().any(|tc| {
                        messages[tail_start..].iter().any(|m| {
                            m.role == Role::Tool
                                && m.tool_call_id.as_deref() == Some(tc.id.as_str())
                        })
                    });
                    if has_result_in_tail {
                        tail_start -= 1;
                    }
                }
            }
        }
    }

    // Ensure no overlap
    if tail_start < head_end {
        tail_start = head_end;
    }

    Boundaries {
        head_end,
        tail_start,
    }
}
