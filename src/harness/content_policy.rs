//! Content-policy guardrail: keeps sexually explicit / NSFW output off the
//! screen entirely, instead of merely masking words.
//!
//! Heuristic, lexicon-based scoring over BOTH text fields (raw, before the
//! profanity masker runs — this guardrail is registered first):
//! - 2+ hits of explicit-sexual terms or severe slurs → the whole utterance
//!   is **blocked** (dropped; the UI shows a "segment withheld" notice)
//! - a single hit → the word is masked in place and the utterance passes
//!
//! Modes (`nsfw_policy` in config.toml): `"block"` (default, behavior above),
//! `"mask"` (never block, always mask), `"off"`.
//!
//! Honest limitation: a lexicon can't judge meaning — explicit content phrased
//! with clean words passes. A local ML classifier can replace this guardrail
//! behind the same trait when that level of rigor is needed.

use super::profanity::{is_strong, mask};
use super::{Guardrail, GuardrailVerdict, UtteranceDraft};

#[derive(Clone, Copy, PartialEq)]
pub enum PolicyMode {
    Block,
    Mask,
}

pub struct ContentPolicy {
    mode: PolicyMode,
}

impl ContentPolicy {
    pub fn new(mode: PolicyMode) -> Self {
        Self { mode }
    }

    pub fn mode_from_str(s: &str) -> Option<PolicyMode> {
        match s {
            "block" => Some(PolicyMode::Block),
            "mask" => Some(PolicyMode::Mask),
            _ => None, // "off" and unknown values → guardrail not registered
        }
    }
}

const EXPLICIT: &[&str] = &[
    "blowjob", "boobs", "cock", "cum", "cumshot", "deepthroat", "dildo",
    "gangbang", "handjob", "hentai", "milf", "orgasm", "porn", "porno",
    "pussy", "threesome", "tits", "xxx",
    // romanized Hindi/Urdu
    "choot", "chudai", "chut", "gaand", "lauda", "lund",
];

fn explicit_hit(lower: &str) -> bool {
    EXPLICIT.contains(&lower) || is_strong(lower)
}

/// Counts policy hits and returns the text with hits masked.
fn scan(text: &str) -> (usize, String) {
    let mut hits = 0;
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    let mut flush = |out: &mut String, word: &mut String| {
        if word.is_empty() {
            return;
        }
        if explicit_hit(&word.to_lowercase()) {
            hits += 1;
            out.push_str(&mask(word, true));
        } else {
            out.push_str(word);
        }
        word.clear();
    };
    for c in text.chars() {
        if c.is_alphabetic() {
            word.push(c);
        } else {
            flush(&mut out, &mut word);
            out.push(c);
        }
    }
    flush(&mut out, &mut word);
    (hits, out)
}

impl Guardrail for ContentPolicy {
    fn name(&self) -> &'static str {
        "content-policy"
    }

    fn check(&self, draft: &mut UtteranceDraft) -> GuardrailVerdict {
        let (orig_hits, orig_masked) = scan(&draft.original_text);
        let (eng_hits, eng_masked) = scan(&draft.english_text);
        let hits = orig_hits.max(eng_hits);

        if hits == 0 {
            return GuardrailVerdict::Pass;
        }
        if self.mode == PolicyMode::Block && hits >= 2 {
            return GuardrailVerdict::Block {
                reason: format!("explicit content ({hits} policy hits)"),
            };
        }
        draft.original_text = orig_masked;
        draft.english_text = eng_masked;
        GuardrailVerdict::Modified
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
            duration_ms: 0,
        }
    }

    #[test]
    fn clean_speech_passes_untouched() {
        let g = ContentPolicy::new(PolicyMode::Block);
        let mut d = draft("let us discuss the quarterly report");
        assert_eq!(g.check(&mut d), GuardrailVerdict::Pass);
        assert_eq!(d.english_text, "let us discuss the quarterly report");
    }

    #[test]
    fn dense_explicit_content_is_blocked_entirely() {
        let g = ContentPolicy::new(PolicyMode::Block);
        let mut d = draft("she wants porn and a blowjob");
        assert!(matches!(g.check(&mut d), GuardrailVerdict::Block { .. }));
    }

    #[test]
    fn a_single_hit_is_masked_not_blocked() {
        let g = ContentPolicy::new(PolicyMode::Block);
        let mut d = draft("he watches porn sometimes");
        assert_eq!(g.check(&mut d), GuardrailVerdict::Modified);
        assert_eq!(d.english_text, "he watches p*** sometimes");
    }

    #[test]
    fn mask_mode_never_blocks() {
        let g = ContentPolicy::new(PolicyMode::Mask);
        let mut d = draft("porn porn porn");
        assert_eq!(g.check(&mut d), GuardrailVerdict::Modified);
        assert_eq!(d.english_text, "p*** p*** p***");
    }

    #[test]
    fn severe_slurs_count_toward_blocking() {
        let g = ContentPolicy::new(PolicyMode::Block);
        let mut d = draft("you cunt, watch that porn");
        assert!(matches!(g.check(&mut d), GuardrailVerdict::Block { .. }));
    }

    #[test]
    fn romanized_hindi_urdu_explicit_terms_are_caught() {
        let g = ContentPolicy::new(PolicyMode::Block);
        let mut d = draft("lund aur chut");
        assert!(matches!(g.check(&mut d), GuardrailVerdict::Block { .. }));
    }
}
