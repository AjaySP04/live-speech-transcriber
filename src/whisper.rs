use anyhow::Result;
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Minimum samples whisper.cpp handles well: 1.1 s @ 16 kHz.
const MIN_SAMPLES: usize = 17_600;

#[derive(Debug, Clone)]
pub struct Transcription {
    pub lang: String,
    pub text: String,
}

pub struct WhisperEngine {
    ctx: WhisperContext,
    threads: i32,
}

pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|s| *s as f32 / 32768.0).collect()
}

impl WhisperEngine {
    pub fn new(model_path: &Path, threads: i32) -> Result<Self> {
        let mut ctx_params = WhisperContextParameters::default();
        // Noticeably faster attention on Metal/GPU backends, no accuracy cost.
        ctx_params.flash_attn(true);
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().expect("model path must be utf-8"),
            ctx_params,
        )?;
        Ok(Self { ctx, threads })
    }

    fn run(&self, audio: &[f32], translate: bool) -> Result<Transcription> {
        let mut padded;
        let audio = if audio.len() < MIN_SAMPLES {
            padded = audio.to_vec();
            padded.resize(MIN_SAMPLES, 0.0);
            &padded[..]
        } else {
            audio
        };

        let mut state = self.ctx.create_state()?;
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some("auto"));
        params.set_translate(translate);
        params.set_n_threads(self.threads);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_no_context(true);
        state.full(params, audio)?;

        // API drift (whisper-rs 0.16): `full_lang_id_from_state` and `full_n_segments`
        // return plain `c_int`, not `Result`, so no `?` here.
        let lang_id = state.full_lang_id_from_state();
        let lang = whisper_rs::get_lang_str(lang_id)
            .unwrap_or("unknown")
            .to_string();

        // API drift: there is no `full_get_segment_text` on `WhisperState`. Segment
        // text is reached via `state.get_segment(i)` -> `WhisperSegment::to_str()`.
        let n = state.full_n_segments();
        let mut text = String::new();
        for i in 0..n {
            if let Some(segment) = state.get_segment(i) {
                text.push_str(segment.to_str()?);
            }
        }
        Ok(Transcription {
            lang,
            text: text.trim().to_string(),
        })
    }

    pub fn transcribe(&self, audio: &[f32]) -> Result<Transcription> {
        self.run(audio, false)
    }

    pub fn translate(&self, audio: &[f32]) -> Result<Transcription> {
        self.run(audio, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::sync::Once;

    static INIT_LOGGING: Once = Once::new();

    fn engine() -> WhisperEngine {
        // Silences whisper.cpp/ggml C-side stderr logging during tests (noop
        // sink since neither the `log_backend` nor `tracing_backend` feature
        // is enabled for whisper-rs in this project).
        INIT_LOGGING.call_once(|| {
            whisper_rs::install_logging_hooks();
        });
        let p = Path::new("models/ggml-tiny.bin");
        assert!(p.exists(), "run scripts/fetch-models.sh tiny");
        WhisperEngine::new(p, 4).unwrap()
    }

    fn fixture_f32() -> Vec<f32> {
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let s: Vec<i16> = r.samples::<i16>().map(|s| s.unwrap()).collect();
        i16_to_f32(&s)
    }

    #[test]
    fn transcribes_and_translates_jfk() {
        let e = engine();
        let audio = fixture_f32();

        let t = e.transcribe(&audio).unwrap();
        assert_eq!(t.lang, "en");
        assert!(t.text.to_lowercase().contains("country"), "got: {}", t.text);

        let tr = e.translate(&audio).unwrap();
        assert!(tr.text.to_lowercase().contains("country"), "got: {}", tr.text);
    }

    #[test]
    fn short_audio_is_padded_not_crashing() {
        let e = engine();
        let audio = vec![0.0f32; 4000]; // 0.25 s of silence
        let t = e.transcribe(&audio).unwrap();
        assert!(t.text.len() < 200); // whatever it hallucinates, it must not crash
    }
}
