//! 13-section structured summary templates for context compression.

/// System prompt for the summarizer LLM.
pub const SUMMARY_SYSTEM_PROMPT: &str = "\
You are a conversation summarizer for an AI coding agent. Produce a structured summary using the EXACT section headers below. The 'Active Task' section is THE SINGLE MOST IMPORTANT — it must clearly state what the agent should resume doing.

Sections:
## Active Task
What the agent should do next. Be specific (file names, function names, exact next step).

## Goal
The overall objective of this conversation.

## Constraints
Any rules, requirements, or limitations mentioned by the user.

## Completed Actions
What was already done (with file paths and outcomes).

## Active State
Current working state (open files, set variables, working directory, installed packages).

## In Progress
Tasks started but not finished.

## Blocked
Tasks that cannot proceed and why.

## Key Decisions
Important choices made. Include reasoning. NEVER summarize this section — carry forward verbatim.

## Resolved Questions
Questions that were asked and answered.

## Pending User Asks
Questions asked to the user that haven't been answered yet.

## Relevant Files
Files read, modified, or created (with brief description of changes).

## Remaining Work
Tasks not yet started.

## Critical Context
Any context that would be lost if not explicitly preserved. NEVER summarize — carry forward verbatim.

Keep your summary concise. Omit sections that have no relevant content.";

/// User prompt template for initial summarization.
///
/// Placeholder: `{turns}` — the formatted conversation turns.
pub const SUMMARY_USER_TEMPLATE: &str = "\
Summarize the following conversation turns:

{turns}

Use the section headers exactly as specified.";

/// User prompt template for iterative (update) summarization.
///
/// Placeholders: `{previous_summary}`, `{new_turns}`.
pub const ITERATIVE_USER_TEMPLATE: &str = "\
Previous summary:
{previous_summary}

New turns since last summary:
{new_turns}

Update the summary to incorporate the new turns. Carry forward the 'Key Decisions' and 'Critical Context' sections verbatim from the previous summary. Update all other sections.";

/// Section headers that must be carried forward verbatim during iterative updates.
pub const CARRY_FORWARD_SECTIONS: &[&str] = &["Key Decisions", "Critical Context"];
