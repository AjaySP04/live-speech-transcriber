use crate::config::Config;
use crate::db::{Db, UtteranceRow};
use crate::harness::{Harness, UtteranceDraft};
use crate::segmenter::{SegmentEvent, Segmenter, SegmenterConfig, CHUNK};
use crate::speakers::{SpeakerEmbedder, SpeakerRegistry};
use crate::vad::Vad;
use crate::whisper::{i16_to_f32, WhisperEngine};
use serde::Serialize;
use std::sync::{Arc, Mutex};

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineEvent {
    SessionStarted {
        session_id: i64,
    },
    SpeechStart,
    Transcribing,
    Utterance {
        speaker: usize,
        lang: String,
        original_text: String,
        english_text: String,
        start_ms: u64,
        duration_ms: u64,
        /// Mean whisper token probability for the translation line (0–1).
        confidence: f32,
        /// Language the translation line is actually in (the configured
        /// target, or "en" when whisper had to fall back).
        translated_to: String,
        /// Milliseconds from utterance-close to this event (translation delay).
        latency_ms: u64,
        /// True when a guardrail withheld the content; the text fields carry
        /// a placeholder ("Sorry — cannot translate this one.") instead.
        blocked: bool,
    },
}

#[derive(Clone)]
pub struct PipelineDeps {
    pub whisper: Arc<WhisperEngine>,
    pub db: Arc<Mutex<Db>>,
    pub cfg: Config,
}

pub fn run_pipeline(
    deps: PipelineDeps,
    session_id: i64,
    audio_rx: std::sync::mpsc::Receiver<Vec<i16>>,
    event_tx: tokio::sync::mpsc::UnboundedSender<PipelineEvent>,
) {
    let _ = event_tx.send(PipelineEvent::SessionStarted { session_id });

    let mut vad = match Vad::new() {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("VAD init failed: {e:#}");
            return;
        }
    };
    let mut embedder = match SpeakerEmbedder::new(&deps.cfg.speaker_model_path()) {
        Ok(e) => e,
        Err(e) => {
            tracing::error!("speaker embedder init failed: {e:#}");
            return;
        }
    };

    let persisted = deps
        .db
        .lock()
        .unwrap()
        .load_speakers(session_id)
        .unwrap_or_default();
    let mut registry = SpeakerRegistry::restore(
        deps.cfg.speaker_similarity_threshold,
        persisted.into_iter().map(|(_, c)| c).collect(),
    );
    let mut first_utterance = deps
        .db
        .lock()
        .unwrap()
        .get_utterances(session_id)
        .map(|u| u.is_empty())
        .unwrap_or(true);

    let harness = Harness::from_config(&deps.cfg);
    let mut segmenter = Segmenter::new(SegmenterConfig {
        vad_threshold: deps.cfg.vad_threshold,
        silence_end_ms: deps.cfg.silence_end_ms,
        pre_roll_ms: deps.cfg.pre_roll_ms,
        min_speech_ms: deps.cfg.min_speech_ms,
        max_utterance_ms: deps.cfg.max_utterance_s * 1000,
    });

    let mut pending: Vec<i16> = Vec::new();

    while let Ok(samples) = audio_rx.recv() {
        pending.extend_from_slice(&samples);
        let mut offset = 0;
        while pending.len() - offset >= CHUNK {
            let chunk = &pending[offset..offset + CHUNK];
            let prob = vad.predict(chunk);
            if let Some(ev) = segmenter.push(chunk, prob) {
                match ev {
                    SegmentEvent::SpeechStart => {
                        let _ = event_tx.send(PipelineEvent::SpeechStart);
                    }
                    SegmentEvent::Utterance {
                        samples,
                        start_ms,
                        duration_ms,
                    } => {
                        let _ = event_tx.send(PipelineEvent::Transcribing);
                        handle_utterance(
                            &deps,
                            &harness,
                            session_id,
                            &samples,
                            start_ms,
                            duration_ms,
                            &mut embedder,
                            &mut registry,
                            &mut first_utterance,
                            &event_tx,
                        );
                    }
                }
            }
            offset += CHUNK;
        }
        pending.drain(..offset);
    }
    tracing::info!("pipeline for session {session_id} finished");
}

#[allow(clippy::too_many_arguments)]
fn handle_utterance(
    deps: &PipelineDeps,
    harness: &Harness,
    session_id: i64,
    samples: &[i16],
    start_ms: u64,
    duration_ms: u64,
    embedder: &mut SpeakerEmbedder,
    registry: &mut SpeakerRegistry,
    first_utterance: &mut bool,
    event_tx: &tokio::sync::mpsc::UnboundedSender<PipelineEvent>,
) {
    // Marks when the finished utterance entered processing; the delta at event
    // time is the translation delay surfaced to the UI and stored in the DB.
    let received = std::time::Instant::now();
    let audio = i16_to_f32(samples);

    let target = deps.cfg.target_lang.clone();
    let (lang, original_text, translation, translated_to, confidence) = if deps.cfg.english_only {
        match deps.whisper.translate(&audio) {
            Ok(t) => (t.lang, String::new(), t.text, "en".to_string(), t.confidence),
            Err(e) => {
                tracing::error!("translate failed: {e:#}");
                return;
            }
        }
    } else {
        // Both passes decode the same audio independently (whisper states share
        // only the immutable model weights), so run them in parallel.
        let (orig, eng) = std::thread::scope(|s| {
            let eng = s.spawn(|| deps.whisper.translate(&audio));
            let orig = deps.whisper.transcribe(&audio);
            (orig, eng.join().expect("translate thread panicked"))
        });
        let orig = match orig {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("transcribe failed: {e:#}");
                return;
            }
        };
        let eng = match eng {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("translate failed: {e:#}");
                return;
            }
        };
        if orig.lang == target && target != "en" {
            // The speaker already speaks the user's language: the native
            // transcription IS the translation line.
            let text = orig.text.clone();
            (orig.lang, orig.text, text, target.clone(), orig.confidence)
        } else {
            // Whisper only translates to English; non-English targets fall
            // back to English (the UI marks the fallback) until an MT model
            // is added.
            (orig.lang, orig.text, eng.text, "en".to_string(), eng.confidence)
        }
    };

    let mut draft = UtteranceDraft {
        lang,
        original_text,
        english_text: translation,
        duration_ms,
    };
    // A blocked utterance still becomes a visible transcript entry — with a
    // placeholder instead of the offending text, which is never stored.
    let blocked = match harness.process(&mut draft) {
        Ok(()) => false,
        Err((guardrail, _reason)) => {
            draft.original_text.clear();
            draft.english_text = match guardrail {
                "language-allowlist" => {
                    "Sorry — cannot translate this one (unsupported language).".to_string()
                }
                _ => "Sorry — cannot translate this one.".to_string(),
            };
            true
        }
    };
    let UtteranceDraft { lang, original_text, english_text, .. } = draft;
    let confidence = if blocked { 0.0 } else { confidence };

    if !blocked && english_text.trim().is_empty() && original_text.trim().is_empty() {
        return; // whisper heard nothing worth keeping
    }

    let speaker = match embedder.embed(&audio) {
        Ok(emb) => {
            let can_create = duration_ms >= deps.cfg.min_new_speaker_ms as u64;
            let num = registry.assign_with_policy(&emb, can_create);
            let _ =
                deps.db
                    .lock()
                    .unwrap()
                    .upsert_speaker(session_id, num as i64, registry.centroid(num));
            num
        }
        Err(e) => {
            tracing::warn!("embedding failed, defaulting speaker 1: {e:#}");
            1
        }
    };

    let latency_ms = received.elapsed().as_millis() as u64;
    let row = UtteranceRow {
        speaker_num: speaker as i64,
        lang: lang.clone(),
        original_text: original_text.clone(),
        english_text: english_text.clone(),
        start_ms: start_ms as i64,
        duration_ms: duration_ms as i64,
        confidence: confidence as f64,
        translated_to: translated_to.clone(),
        latency_ms: latency_ms as i64,
        blocked,
    };
    if let Err(e) = deps.db.lock().unwrap().insert_utterance(session_id, &row) {
        tracing::error!("db insert failed: {e:#}");
    }

    if *first_utterance && !blocked {
        let title: String = english_text
            .split_whitespace()
            .take(8)
            .collect::<Vec<_>>()
            .join(" ");
        let _ = deps.db.lock().unwrap().set_session_title(session_id, &title);
        *first_utterance = false;
    }

    let _ = event_tx.send(PipelineEvent::Utterance {
        speaker,
        lang,
        original_text,
        english_text,
        start_ms,
        duration_ms,
        confidence,
        translated_to,
        latency_ms,
        blocked,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::db::Db;
    use crate::whisper::WhisperEngine;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    #[test]
    fn jfk_through_pipeline_yields_english_utterance() {
        assert!(
            Path::new("models/ggml-tiny.bin").exists(),
            "run scripts/fetch-models.sh tiny"
        );
        let mut cfg = Config::default();
        cfg.whisper_model = "tiny".into();

        let whisper = Arc::new(WhisperEngine::new(&cfg.whisper_model_path(), 4).unwrap());
        let db = Arc::new(Mutex::new(Db::open_in_memory().unwrap()));
        let session_id = db.lock().unwrap().create_session().unwrap();

        let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<i16>>();
        let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel();

        let deps = PipelineDeps {
            whisper,
            db: db.clone(),
            cfg,
        };
        let handle = std::thread::spawn(move || run_pipeline(deps, session_id, audio_rx, event_tx));

        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let samples: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
        for chunk in samples.chunks(4096) {
            audio_tx.send(chunk.to_vec()).unwrap();
        }
        // a second of silence so the segmenter closes the last utterance
        for _ in 0..8 {
            audio_tx.send(vec![0i16; 4096]).unwrap();
        }
        drop(audio_tx);
        handle.join().unwrap();

        let mut utterances = Vec::new();
        while let Ok(ev) = event_rx.try_recv() {
            if let PipelineEvent::Utterance {
                english_text,
                speaker,
                ..
            } = ev
            {
                utterances.push((speaker, english_text));
            }
        }
        assert!(!utterances.is_empty(), "no utterances produced");
        let all_text: String = utterances.iter().map(|(_, t)| t.to_lowercase()).collect();
        assert!(all_text.contains("country"), "got: {all_text}");
        assert!(
            utterances.iter().all(|(s, _)| *s == 1),
            "one voice must be one speaker"
        );

        let rows = db.lock().unwrap().get_utterances(session_id).unwrap();
        assert_eq!(rows.len(), utterances.len());
        let sessions = db.lock().unwrap().list_sessions().unwrap();
        assert!(!sessions[0].title.is_empty(), "title should be auto-set");
    }
}
