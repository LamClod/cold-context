//! Token budget allocation for context window management.

/// Configuration for how the context window budget is divided.
#[derive(Debug, Clone)]
pub struct BudgetConfig {
    /// Fraction of context reserved for system prompt (default 0.15).
    pub system_prompt_percent: f32,
    /// Fraction of context reserved for tool definitions (default 0.10).
    pub tool_definitions_percent: f32,
    /// Fraction of context reserved for completion output (default 0.15).
    pub completion_percent: f32,
    // Conversation gets the rest (default 0.60).
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            system_prompt_percent: 0.15,
            tool_definitions_percent: 0.10,
            completion_percent: 0.15,
        }
    }
}

/// Computed token budget for each section of the context window.
#[derive(Debug, Clone)]
pub struct TokenBudget {
    /// Total context length in tokens.
    pub total: u32,
    /// Tokens reserved for the system prompt.
    pub system_prompt: u32,
    /// Tokens reserved for tool definitions.
    pub tool_definitions: u32,
    /// Tokens reserved for completion output.
    pub completion: u32,
    /// Tokens available for conversation history.
    pub conversation: u32,
}

impl TokenBudget {
    /// Compute a token budget from total context length and allocation config.
    #[must_use]
    pub fn new(context_length: u32, config: &BudgetConfig) -> Self {
        let total = context_length;
        let system_prompt = to_tokens(total, config.system_prompt_percent);
        let tool_definitions = to_tokens(total, config.tool_definitions_percent);
        let completion = to_tokens(total, config.completion_percent);
        let conversation = total.saturating_sub(system_prompt + tool_definitions + completion);

        Self {
            total,
            system_prompt,
            tool_definitions,
            completion,
            conversation,
        }
    }

    /// How many conversation tokens remain given current prompt usage.
    ///
    /// Returns a negative value if over budget.
    #[must_use]
    pub fn conversation_remaining(&self, current_prompt_tokens: u32) -> i32 {
        i32::try_from(self.conversation).unwrap_or(i32::MAX)
            - i32::try_from(current_prompt_tokens).unwrap_or(i32::MAX)
    }

    /// Whether the current prompt tokens exceed the conversation budget.
    #[must_use]
    pub const fn is_conversation_over_budget(&self, current_prompt_tokens: u32) -> bool {
        current_prompt_tokens > self.conversation
    }
}

/// Convert a fraction of total tokens to an integer count.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn to_tokens(total: u32, percent: f32) -> u32 {
    (f64::from(total) * f64::from(percent)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_budget_60_percent_conversation() {
        let config = BudgetConfig::default();
        let budget = TokenBudget::new(100_000, &config);
        // 15% + 10% + 15% = 40% reserved, 60% for conversation
        assert_eq!(budget.system_prompt, 15_000);
        assert_eq!(budget.tool_definitions, 10_000);
        assert_eq!(budget.completion, 15_000);
        assert_eq!(budget.conversation, 60_000);
    }

    #[test]
    fn conversation_remaining_positive() {
        let budget = TokenBudget::new(100_000, &BudgetConfig::default());
        assert_eq!(budget.conversation_remaining(30_000), 30_000);
    }

    #[test]
    fn conversation_remaining_negative() {
        let budget = TokenBudget::new(100_000, &BudgetConfig::default());
        assert_eq!(budget.conversation_remaining(70_000), -10_000);
    }

    #[test]
    fn over_budget_check() {
        let budget = TokenBudget::new(100_000, &BudgetConfig::default());
        assert!(!budget.is_conversation_over_budget(50_000));
        assert!(budget.is_conversation_over_budget(70_000));
    }
}
