//! Profanity guardrail: masks abusive language in transcript text.
//!
//! Matching is case-insensitive on whole words. Two tiers:
//! - **mild** terms keep their first and last letter ("f**k")
//! - **strong** terms (slurs, severe abuse) keep only their first letter ("c***")
//!
//! The lists deliberately include common romanized Hindi/Urdu abuse, since
//! Whisper often carries those words into the English translation verbatim.
//! Disable with `profanity_filter = false` in config.toml.

use super::{Guardrail, GuardrailVerdict, UtteranceDraft};

pub struct ProfanityGuardrail;

impl Guardrail for ProfanityGuardrail {
    fn name(&self) -> &'static str {
        "profanity"
    }

    fn check(&self, draft: &mut UtteranceDraft) -> GuardrailVerdict {
        let original = clean(&draft.original_text);
        let english = clean(&draft.english_text);
        let modified = original != draft.original_text || english != draft.english_text;
        draft.original_text = original;
        draft.english_text = english;
        if modified {
            GuardrailVerdict::Modified
        } else {
            GuardrailVerdict::Pass
        }
    }
}

const MILD: &[&str] = &[
    "arse", "arsehole", "asshole", "bastard", "bitch", "bloody", "bullshit",
    "crap", "dick", "dickhead", "fuck", "fucked", "fucker", "fucking", "piss",
    "prick", "shit", "shitty", "wanker",
    // romanized Hindi/Urdu
    "behenchod", "bhenchod", "bhosdike", "chutiya", "gandu", "harami",
    "kamina", "kutta", "kutte", "madarchod", "saala", "saale",
];

const STRONG: &[&str] = &[
    "chink", "cunt", "fag", "faggot", "kike", "motherfucker", "nigga",
    "nigger", "paki", "randi", "retard", "slut", "spic", "tranny", "whore",
];

pub(super) fn is_strong(lower: &str) -> bool {
    STRONG.contains(&lower)
}

pub(super) fn mask(word: &str, strong: bool) -> String {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    if n <= 2 {
        return "*".repeat(n);
    }
    if strong || n <= 3 {
        let mut out = String::new();
        out.push(chars[0]);
        out.extend(std::iter::repeat('*').take(n - 1));
        out
    } else {
        let mut out = String::new();
        out.push(chars[0]);
        out.extend(std::iter::repeat('*').take(n - 2));
        out.push(chars[n - 1]);
        out
    }
}

fn classify(lower: &str) -> Option<bool> {
    if STRONG.contains(&lower) {
        Some(true)
    } else if MILD.contains(&lower) {
        Some(false)
    } else {
        None
    }
}

/// Returns `text` with abusive words masked. Non-word characters and clean
/// words pass through untouched.
pub fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    for c in text.chars() {
        if c.is_alphabetic() {
            word.push(c);
        } else {
            flush_word(&mut out, &mut word);
            out.push(c);
        }
    }
    flush_word(&mut out, &mut word);
    out
}

fn flush_word(out: &mut String, word: &mut String) {
    if word.is_empty() {
        return;
    }
    let lower = word.to_lowercase();
    match classify(&lower) {
        Some(strong) => out.push_str(&mask(word, strong)),
        None => out.push_str(word),
    }
    word.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text_is_untouched() {
        let s = "How are you? आप कैसे हैं؟ Everything is fine.";
        assert_eq!(clean(s), s);
    }

    #[test]
    fn mild_words_keep_first_and_last_letter() {
        assert_eq!(clean("what the fuck"), "what the f**k");
        assert_eq!(clean("this is bullshit."), "this is b******t.");
    }

    #[test]
    fn strong_words_keep_only_first_letter() {
        assert_eq!(clean("you cunt"), "you c***");
    }

    #[test]
    fn matching_is_case_insensitive_and_preserves_case() {
        assert_eq!(clean("Fuck!"), "F**k!");
        assert_eq!(clean("FUCK"), "F**K");
    }

    #[test]
    fn word_boundaries_are_respected() {
        // "class", "assist", "Scunthorpe" must not trigger substring matches
        let s = "The class assistant from Scunthorpe passed.";
        assert_eq!(clean(s), s);
    }

    #[test]
    fn romanized_hindi_urdu_abuse_is_masked() {
        assert_eq!(clean("arre chutiya hai kya"), "arre c*****a hai kya");
        assert_eq!(clean("Madarchod bola usne"), "M*******d bola usne");
    }

    #[test]
    fn verdict_reflects_whether_text_changed() {
        let g = ProfanityGuardrail;
        let mut clean_draft = UtteranceDraft {
            lang: "en".into(),
            original_text: String::new(),
            english_text: "all good here".into(),
            duration_ms: 0,
        };
        assert_eq!(g.check(&mut clean_draft), GuardrailVerdict::Pass);

        let mut dirty = UtteranceDraft {
            lang: "en".into(),
            original_text: String::new(),
            english_text: "oh shit".into(),
            duration_ms: 0,
        };
        assert_eq!(g.check(&mut dirty), GuardrailVerdict::Modified);
        assert_eq!(dirty.english_text, "oh s**t");
    }
}
