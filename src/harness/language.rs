//! Language allowlist guardrail: drops utterances in languages the app is
//! not tuned for. Narrowing the accepted set (English, Hindi, Urdu, Arabic
//! by default) avoids confidently-wrong transcriptions of unsupported speech.

use super::{Guardrail, GuardrailVerdict, UtteranceDraft};

pub struct LanguageAllowlist {
    allowed: Vec<String>,
}

impl LanguageAllowlist {
    pub fn new(allowed: Vec<String>) -> Self {
        Self { allowed }
    }
}

impl Guardrail for LanguageAllowlist {
    fn name(&self) -> &'static str {
        "language-allowlist"
    }

    fn check(&self, draft: &mut UtteranceDraft) -> GuardrailVerdict {
        if self.allowed.iter().any(|l| l == &draft.lang) {
            GuardrailVerdict::Pass
        } else {
            GuardrailVerdict::Block {
                reason: format!("detected language '{}' is not in the allowlist", draft.lang),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(lang: &str) -> UtteranceDraft {
        UtteranceDraft {
            lang: lang.into(),
            original_text: "x".into(),
            english_text: "x".into(),
            duration_ms: 0,
        }
    }

    #[test]
    fn allowed_languages_pass() {
        let g = LanguageAllowlist::new(vec!["en".into(), "hi".into()]);
        assert_eq!(g.check(&mut draft("hi")), GuardrailVerdict::Pass);
        assert_eq!(g.check(&mut draft("en")), GuardrailVerdict::Pass);
    }

    #[test]
    fn other_languages_are_blocked() {
        let g = LanguageAllowlist::new(vec!["en".into(), "hi".into(), "ur".into(), "ar".into()]);
        assert!(matches!(
            g.check(&mut draft("fr")),
            GuardrailVerdict::Block { .. }
        ));
        assert!(matches!(
            g.check(&mut draft("unknown")),
            GuardrailVerdict::Block { .. }
        ));
    }
}
