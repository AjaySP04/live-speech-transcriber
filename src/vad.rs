use anyhow::Result;
use voice_activity_detector::VoiceActivityDetector;

/// Silero V5 VAD. 16 kHz, 512-sample chunks.
pub struct Vad {
    inner: VoiceActivityDetector,
}

impl Vad {
    pub fn new() -> Result<Self> {
        let inner = VoiceActivityDetector::builder()
            .sample_rate(16000)
            .chunk_size(512usize)
            .build()?;
        Ok(Self { inner })
    }

    pub fn predict(&mut self, chunk: &[i16]) -> f32 {
        self.inner.predict(chunk.iter().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_fixture() -> Vec<i16> {
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        assert_eq!(r.spec().sample_rate, 16000);
        r.samples::<i16>().map(|s| s.unwrap()).collect()
    }

    #[test]
    fn speech_scores_high_silence_scores_low() {
        let mut vad = Vad::new().unwrap();
        let samples = read_fixture();
        let max_speech = samples
            .chunks_exact(512)
            .map(|c| vad.predict(c))
            .fold(0.0f32, f32::max);
        assert!(max_speech > 0.8, "expected speech in jfk.wav, max prob {max_speech}");

        let mut vad2 = Vad::new().unwrap();
        let silence = vec![0i16; 512];
        let p: f32 = (0..10).map(|_| vad2.predict(&silence)).fold(0.0, f32::max);
        assert!(p < 0.3, "silence scored {p}");
    }
}
