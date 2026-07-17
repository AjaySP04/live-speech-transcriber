use std::collections::VecDeque;

pub const CHUNK: usize = 512;
pub const CHUNK_MS: u64 = 32;

#[derive(Debug, PartialEq, Clone)]
pub enum SegmentEvent {
    SpeechStart,
    Utterance {
        samples: Vec<i16>,
        start_ms: u64,
        duration_ms: u64,
    },
}

#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    pub vad_threshold: f32,
    pub silence_end_ms: u32,
    pub pre_roll_ms: u32,
    pub min_speech_ms: u32,
    pub max_utterance_ms: u32,
}

enum State {
    Idle,
    Speaking,
}

pub struct Segmenter {
    cfg: SegmenterConfig,
    state: State,
    pre_roll: VecDeque<Vec<i16>>,
    buf: Vec<i16>,
    buf_start_chunk: u64,
    silence_chunks: u32,
    speech_chunks: u32,
    chunk_index: u64,
}

impl Segmenter {
    pub fn new(cfg: SegmenterConfig) -> Self {
        Self {
            cfg,
            state: State::Idle,
            pre_roll: VecDeque::new(),
            buf: Vec::new(),
            buf_start_chunk: 0,
            silence_chunks: 0,
            speech_chunks: 0,
            chunk_index: 0,
        }
    }

    fn pre_roll_cap(&self) -> usize {
        (self.cfg.pre_roll_ms as u64 / CHUNK_MS).max(1) as usize
    }

    fn silence_limit(&self) -> u32 {
        (self.cfg.silence_end_ms as u64 / CHUNK_MS).max(1) as u32
    }

    fn min_speech(&self) -> u32 {
        (self.cfg.min_speech_ms as u64 / CHUNK_MS).max(1) as u32
    }

    fn max_chunks(&self) -> usize {
        (self.cfg.max_utterance_ms as u64 / CHUNK_MS).max(2) as usize
    }

    fn take_utterance(&mut self) -> SegmentEvent {
        let samples = std::mem::take(&mut self.buf);
        let chunks = (samples.len() / CHUNK) as u64;
        SegmentEvent::Utterance {
            samples,
            start_ms: self.buf_start_chunk * CHUNK_MS,
            duration_ms: chunks * CHUNK_MS,
        }
    }

    /// Feed exactly one 512-sample chunk with its VAD probability.
    pub fn push(&mut self, chunk: &[i16], prob: f32) -> Option<SegmentEvent> {
        debug_assert_eq!(chunk.len(), CHUNK);
        let idx = self.chunk_index;
        self.chunk_index += 1;
        let speech = prob >= self.cfg.vad_threshold;

        match self.state {
            State::Idle => {
                self.pre_roll.push_back(chunk.to_vec());
                while self.pre_roll.len() > self.pre_roll_cap() {
                    self.pre_roll.pop_front();
                }
                if speech {
                    self.state = State::Speaking;
                    self.buf_start_chunk = idx + 1 - self.pre_roll.len() as u64;
                    self.buf = self.pre_roll.drain(..).flatten().collect();
                    self.silence_chunks = 0;
                    self.speech_chunks = 1;
                    Some(SegmentEvent::SpeechStart)
                } else {
                    None
                }
            }
            State::Speaking => {
                self.buf.extend_from_slice(chunk);
                if speech {
                    self.silence_chunks = 0;
                    self.speech_chunks += 1;
                } else {
                    self.silence_chunks += 1;
                }

                if self.buf.len() / CHUNK >= self.max_chunks() {
                    let ev = self.take_utterance();
                    // stay Speaking; next utterance continues from here
                    self.buf_start_chunk = idx + 1;
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    return Some(ev);
                }

                if self.silence_chunks >= self.silence_limit() {
                    let enough = self.speech_chunks >= self.min_speech();
                    let ev = if enough { Some(self.take_utterance()) } else { None };
                    self.buf.clear();
                    self.state = State::Idle;
                    self.pre_roll.clear();
                    self.silence_chunks = 0;
                    self.speech_chunks = 0;
                    return ev;
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            vad_threshold: 0.5,
            silence_end_ms: 320,  // 10 chunks
            pre_roll_ms: 96,      // 3 chunks
            min_speech_ms: 160,   // 5 chunks
            max_utterance_ms: 3200, // 100 chunks
        }
    }

    fn push_n(s: &mut Segmenter, n: usize, prob: f32) -> Vec<SegmentEvent> {
        let chunk = [100i16; CHUNK];
        (0..n).filter_map(|_| s.push(&chunk, prob)).collect()
    }

    #[test]
    fn silence_produces_no_events() {
        let mut s = Segmenter::new(cfg());
        assert!(push_n(&mut s, 50, 0.1).is_empty());
    }

    #[test]
    fn speech_then_silence_emits_utterance_with_preroll() {
        let mut s = Segmenter::new(cfg());
        push_n(&mut s, 3, 0.1); // fills pre-roll
        let ev = push_n(&mut s, 20, 0.9); // speech
        assert_eq!(ev, vec![SegmentEvent::SpeechStart]);
        let ev = push_n(&mut s, 10, 0.1); // silence closes it
        match &ev[0] {
            SegmentEvent::Utterance { samples, start_ms, duration_ms } => {
                // 3 pre-roll (incl. current trigger chunk counted once) + 19 more speech + 10 silence chunks buffered
                assert_eq!(samples.len() % CHUNK, 0);
                let chunks = samples.len() / CHUNK;
                assert_eq!(chunks, 3 + 19 + 10);
                // pre-roll cap is 3: trigger chunk pushes out chunk 0, so buffer starts at chunk 1
                assert_eq!(*start_ms, CHUNK_MS);
                assert_eq!(*duration_ms, (chunks as u64) * CHUNK_MS);
            }
            other => panic!("expected Utterance, got {other:?}"),
        }
    }

    #[test]
    fn short_blip_is_discarded() {
        let mut s = Segmenter::new(cfg());
        let ev = push_n(&mut s, 2, 0.9); // only 2 speech chunks < min 5
        assert_eq!(ev, vec![SegmentEvent::SpeechStart]);
        let ev = push_n(&mut s, 10, 0.1);
        assert!(ev.is_empty(), "blip must be discarded, got {ev:?}");
    }

    #[test]
    fn long_speech_force_flushes_at_max() {
        let mut s = Segmenter::new(cfg());
        let ev = push_n(&mut s, 150, 0.9); // exceeds 100-chunk cap
        let utts: Vec<_> = ev.iter().filter(|e| matches!(e, SegmentEvent::Utterance { .. })).collect();
        assert_eq!(utts.len(), 1);
        if let SegmentEvent::Utterance { samples, .. } = utts[0] {
            assert_eq!(samples.len() / CHUNK, 100);
        }
    }
}
