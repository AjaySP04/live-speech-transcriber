use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    pub port: u16,
    pub models_dir: String,
    pub data_dir: String,
    pub whisper_model: String,
    pub english_only: bool,
    pub vad_threshold: f32,
    pub silence_end_ms: u32,
    pub pre_roll_ms: u32,
    pub min_speech_ms: u32,
    pub max_utterance_s: u32,
    pub speaker_similarity_threshold: f32,
    pub min_new_speaker_ms: u32,
    pub whisper_threads: i32,
    pub tls: bool,
    pub profanity_filter: bool,
    /// NSFW/abuse policy: "block" (drop dense explicit content, mask single
    /// hits), "mask" (never drop, always mask), or "off".
    pub nsfw_policy: String,
    /// BCP-ish whisper language codes the pipeline accepts; anything else is dropped.
    pub allowed_languages: Vec<String>,
    /// Language the translation line is rendered in. Whisper can only translate
    /// to English, so non-"en" targets show native transcription when the
    /// speaker already speaks the target language and fall back to English
    /// otherwise (until a dedicated MT model lands).
    pub target_lang: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: 8080,
            models_dir: "models".into(),
            data_dir: "data".into(),
            whisper_model: "medium".into(),
            english_only: false,
            vad_threshold: 0.5,
            silence_end_ms: 700,
            pre_roll_ms: 300,
            min_speech_ms: 250,
            max_utterance_s: 25,
            speaker_similarity_threshold: 0.65,
            min_new_speaker_ms: 2000,
            whisper_threads: 4,
            tls: false,
            profanity_filter: true,
            nsfw_policy: "block".into(),
            allowed_languages: vec!["en".into(), "hi".into(), "ur".into(), "ar".into()],
            target_lang: "en".into(),
        }
    }
}

impl Config {
    /// Loads config from a TOML file; missing file = all defaults.
    /// Env overrides: LT_PORT, LT_WHISPER_MODEL, LT_ENGLISH_ONLY.
    pub fn load(path: &str) -> Result<Config> {
        let mut cfg: Config = match std::fs::read_to_string(path) {
            Ok(s) => toml::from_str(&s)?,
            Err(_) => Config::default(),
        };
        if let Ok(v) = std::env::var("LT_PORT") {
            cfg.port = v.parse()?;
        }
        if let Ok(v) = std::env::var("LT_WHISPER_MODEL") {
            cfg.whisper_model = v;
        }
        if let Ok(v) = std::env::var("LT_ENGLISH_ONLY") {
            cfg.english_only = v == "1" || v.eq_ignore_ascii_case("true");
        }
        Ok(cfg)
    }

    pub fn whisper_model_path(&self) -> PathBuf {
        Path::new(&self.models_dir).join(format!("ggml-{}.bin", self.whisper_model))
    }

    pub fn speaker_model_path(&self) -> PathBuf {
        Path::new(&self.models_dir).join("nemo_en_titanet_small.onnx")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_gives_defaults() {
        let cfg = Config::load("/nonexistent/config.toml").unwrap();
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.whisper_model, "medium");
        assert!(!cfg.english_only);
        assert_eq!(cfg.whisper_model_path().to_str().unwrap(), "models/ggml-medium.bin");
    }

    #[test]
    fn partial_toml_overrides_defaults() {
        let dir = std::env::temp_dir().join("lt-config-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.toml");
        std::fs::write(&p, "whisper_model = \"tiny\"\nport = 9999\n").unwrap();
        let cfg = Config::load(p.to_str().unwrap()).unwrap();
        assert_eq!(cfg.port, 9999);
        assert_eq!(cfg.whisper_model, "tiny");
        assert_eq!(cfg.silence_end_ms, 700); // untouched default
    }
}
