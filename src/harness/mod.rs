//! The AI harness: everything that runs *between* raw transcription output
//! and what the app stores/displays.
//!
//! Two extension seams, executed by [`Harness::process`] on every utterance:
//!
//! - [`Guardrail`] — inspect/rewrite/block an utterance before it leaves the
//!   pipeline (content policy, PII scrubbing, quality gates, …). Guardrails
//!   run in registration order; a `Block` verdict short-circuits and the
//!   utterance is dropped (never stored, never sent to the client).
//! - [`Tool`] — post-transcript actions triggered by utterance content
//!   (voice commands, webhooks, summarizers, LLM calls). The seam exists but
//!   no tools ship yet; `matches`/`invoke` run best-effort after guardrails
//!   pass and can never block or mutate the transcript.
//!
//! To add a guardrail or tool: implement the trait in a new file in this
//! module, then register it in [`Harness::from_config`].

pub mod content_policy;
pub mod language;
pub mod profanity;

use crate::config::Config;

/// An utterance after transcription/translation, before persistence.
/// Guardrails may rewrite the text fields in place.
#[derive(Debug, Clone)]
pub struct UtteranceDraft {
    pub lang: String,
    pub original_text: String,
    pub english_text: String,
    pub duration_ms: u64,
}

#[derive(Debug, PartialEq)]
pub enum GuardrailVerdict {
    /// Untouched.
    Pass,
    /// Text fields were rewritten (e.g. masked); processing continues.
    Modified,
    /// Drop the utterance entirely. `reason` is logged, never shown to users.
    Block { reason: String },
}

pub trait Guardrail: Send + Sync {
    fn name(&self) -> &'static str;
    fn check(&self, draft: &mut UtteranceDraft) -> GuardrailVerdict;
}

/// Future tool-calling seam. Intentionally has no implementations yet.
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// Cheap predicate deciding whether this tool should run for `draft`.
    fn matches(&self, draft: &UtteranceDraft) -> bool;
    /// Side-effectful action. Errors are logged, never fatal to the pipeline.
    fn invoke(&self, draft: &UtteranceDraft) -> anyhow::Result<()>;
}

pub struct Harness {
    guardrails: Vec<Box<dyn Guardrail>>,
    tools: Vec<Box<dyn Tool>>,
}

impl Harness {
    /// Assembles the harness from config. Every guardrail/tool is registered
    /// here and nowhere else.
    pub fn from_config(cfg: &Config) -> Self {
        let mut guardrails: Vec<Box<dyn Guardrail>> = Vec::new();
        if !cfg.allowed_languages.is_empty() {
            guardrails.push(Box::new(language::LanguageAllowlist::new(
                cfg.allowed_languages.clone(),
            )));
        }
        // Content policy must run before the profanity masker so it scores
        // the raw text (masked words wouldn't count toward blocking).
        if let Some(mode) = content_policy::ContentPolicy::mode_from_str(&cfg.nsfw_policy) {
            guardrails.push(Box::new(content_policy::ContentPolicy::new(mode)));
        }
        if cfg.profanity_filter {
            guardrails.push(Box::new(profanity::ProfanityGuardrail));
        }
        Self { guardrails, tools: Vec::new() }
    }

    #[cfg(test)]
    pub fn with_parts(guardrails: Vec<Box<dyn Guardrail>>, tools: Vec<Box<dyn Tool>>) -> Self {
        Self { guardrails, tools }
    }

    /// Runs the utterance through all guardrails, then fires matching tools.
    /// Returns `Err((guardrail_name, reason))` when the utterance was blocked.
    pub fn process(&self, draft: &mut UtteranceDraft) -> Result<(), (&'static str, String)> {
        for g in &self.guardrails {
            match g.check(draft) {
                GuardrailVerdict::Pass => {}
                GuardrailVerdict::Modified => {
                    tracing::debug!(guardrail = g.name(), "utterance modified");
                }
                GuardrailVerdict::Block { reason } => {
                    tracing::info!(guardrail = g.name(), reason, "utterance blocked");
                    return Err((g.name(), reason));
                }
            }
        }
        for t in &self.tools {
            if t.matches(draft) {
                if let Err(e) = t.invoke(draft) {
                    tracing::warn!(tool = t.name(), "tool invocation failed: {e:#}");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draft(text: &str) -> UtteranceDraft {
        UtteranceDraft {
            lang: "en".into(),
            original_text: String::new(),
            english_text: text.into(),
            duration_ms: 1000,
        }
    }

    struct BlockEverything;
    impl Guardrail for BlockEverything {
        fn name(&self) -> &'static str {
            "block-everything"
        }
        fn check(&self, _d: &mut UtteranceDraft) -> GuardrailVerdict {
            GuardrailVerdict::Block { reason: "test".into() }
        }
    }

    #[test]
    fn empty_harness_passes_everything() {
        let h = Harness::with_parts(vec![], vec![]);
        let mut d = draft("anything at all");
        assert!(h.process(&mut d).is_ok());
        assert_eq!(d.english_text, "anything at all");
    }

    #[test]
    fn profanity_guardrail_rewrites_in_place() {
        let h = Harness::with_parts(vec![Box::new(profanity::ProfanityGuardrail)], vec![]);
        let mut d = draft("what the fuck");
        assert!(h.process(&mut d).is_ok());
        assert_eq!(d.english_text, "what the f**k");
    }

    #[test]
    fn a_blocking_guardrail_short_circuits() {
        let h = Harness::with_parts(
            vec![Box::new(BlockEverything), Box::new(profanity::ProfanityGuardrail)],
            vec![],
        );
        let mut d = draft("hello");
        assert_eq!(h.process(&mut d), Err(("block-everything", "test".to_string())));
    }

    #[test]
    fn from_config_respects_the_filter_flag() {
        let mut cfg = crate::config::Config::default();
        cfg.profanity_filter = false;
        let h = Harness::from_config(&cfg);
        let mut d = draft("bullshit");
        assert!(h.process(&mut d).is_ok());
        assert_eq!(d.english_text, "bullshit", "disabled filter must not rewrite");
    }
}
