//! Sensitive information redaction for conversation content.
//!
//! Detects and replaces API keys, tokens, private keys, URL credentials,
//! and environment-style secrets.

use regex::Regex;
use std::sync::LazyLock;

struct RedactionPattern {
    regex: Regex,
    replacement: &'static str,
}

static PATTERNS: LazyLock<Vec<RedactionPattern>> = LazyLock::new(|| {
    vec![
        // Private key blocks (must come before single-line patterns)
        RedactionPattern {
            regex: Regex::new(
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            )
            .unwrap(),
            replacement: "[REDACTED:private_key]",
        },
        // JWT tokens
        RedactionPattern {
            regex: Regex::new(r"eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+").unwrap(),
            replacement: "[REDACTED:jwt]",
        },
        // GitHub tokens
        RedactionPattern {
            regex: Regex::new(r"gh[ps]_[a-zA-Z0-9]{36,}").unwrap(),
            replacement: "[REDACTED:github_token]",
        },
        // AWS keys
        RedactionPattern {
            regex: Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
            replacement: "[REDACTED:aws_key]",
        },
        // API keys (sk-...)
        RedactionPattern {
            regex: Regex::new(r"sk-[a-zA-Z0-9]{20,}").unwrap(),
            replacement: "[REDACTED:api_key]",
        },
        // URL passwords (://user:pass@)
        RedactionPattern {
            regex: Regex::new(r"://[^:]+:[^@]+@").unwrap(),
            replacement: "://[REDACTED]@",
        },
        // Env-style secrets
        RedactionPattern {
            regex: Regex::new(r"(?i)(PASSWORD|SECRET|TOKEN|API_KEY)\s*=\s*\S+").unwrap(),
            replacement: "$1=[REDACTED]",
        },
    ]
});

/// Replace sensitive patterns in `text` with redaction markers.
#[must_use]
pub fn redact_sensitive(text: &str) -> String {
    let mut result = text.to_string();
    for pattern in PATTERNS.iter() {
        result = pattern
            .regex
            .replace_all(&result, pattern.replacement)
            .into_owned();
    }
    result
}

/// Check whether `text` contains any sensitive patterns.
#[must_use]
pub fn has_sensitive_content(text: &str) -> bool {
    PATTERNS.iter().any(|p| p.regex.is_match(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_api_key() {
        let input = "my key is sk-abc123def456ghi789jkl012";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED:api_key]"));
        assert!(!output.contains("sk-abc123"));
    }

    #[test]
    fn redacts_jwt() {
        let input = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED:jwt]"));
    }

    #[test]
    fn redacts_github_token() {
        let input = "token is ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl here";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED:github_token]"));
        assert!(!output.contains("ghp_ABCDEF"));
    }

    #[test]
    fn redacts_aws_key() {
        let input = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED:aws_key]"));
    }

    #[test]
    fn redacts_private_key() {
        let input =
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED:private_key]"));
    }

    #[test]
    fn redacts_url_password() {
        let input = "postgres://admin:supersecret@db.example.com/mydb";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED]@"));
        assert!(!output.contains("supersecret"));
    }

    #[test]
    fn redacts_env_secrets() {
        let input = "PASSWORD=hunter2\nSECRET=mysecret";
        let output = redact_sensitive(input);
        assert!(output.contains("[REDACTED]"));
        assert!(!output.contains("hunter2"));
        assert!(!output.contains("mysecret"));
    }

    #[test]
    fn clean_text_unchanged() {
        let input = "This is normal conversation text with no secrets.";
        assert_eq!(redact_sensitive(input), input);
        assert!(!has_sensitive_content(input));
    }

    #[test]
    fn has_sensitive_detects() {
        assert!(has_sensitive_content("key=sk-abcdefghijklmnopqrstuvwxyz"));
        assert!(!has_sensitive_content("just normal text"));
    }
}
