use anyhow::Result;
use sherpa_rs::speaker_id::{EmbeddingExtractor, ExtractorConfig};
use std::path::Path;

pub struct SpeakerEmbedder {
    inner: EmbeddingExtractor,
}

impl SpeakerEmbedder {
    pub fn new(model_path: &Path) -> Result<Self> {
        let config = ExtractorConfig {
            model: model_path.to_string_lossy().into_owned(),
            provider: None,
            num_threads: Some(2),
            debug: false,
        };
        // API drift: sherpa-rs 0.6.8 returns `eyre::Result`, whose error type
        // (`eyre::Report`) does not implement `std::error::Error`, so `?`
        // cannot auto-convert it into `anyhow::Error`. Stringify instead.
        let inner = EmbeddingExtractor::new(config).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        self.inner
            .compute_speaker_embedding(samples.to_vec(), 16000)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Online speaker clustering: cosine similarity against running centroids.
pub struct SpeakerRegistry {
    threshold: f32,
    speakers: Vec<(Vec<f32>, u32)>, // (centroid, sample count)
}

impl SpeakerRegistry {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            speakers: Vec::new(),
        }
    }

    /// Restore from persisted centroids ordered by speaker number.
    pub fn restore(threshold: f32, centroids: Vec<Vec<f32>>) -> Self {
        Self {
            threshold,
            speakers: centroids.into_iter().map(|c| (c, 1)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    pub fn centroid(&self, num: usize) -> &[f32] {
        &self.speakers[num - 1].0
    }

    /// Returns the 1-based speaker number, creating a new speaker if nothing matches.
    pub fn assign(&mut self, emb: &[f32]) -> usize {
        self.assign_with_policy(emb, true)
    }

    /// Like `assign`, but when `can_create` is false a below-threshold embedding
    /// is attributed to the nearest existing speaker WITHOUT updating its centroid
    /// (short audio embeds unreliably; don't pollute centroids or spawn phantom speakers).
    /// An empty registry always creates speaker 1.
    pub fn assign_with_policy(&mut self, emb: &[f32], can_create: bool) -> usize {
        let mut best: Option<(usize, f32)> = None;
        for (i, (c, _)) in self.speakers.iter().enumerate() {
            let sim = cosine(c, emb);
            if best.map_or(true, |(_, b)| sim > b) {
                best = Some((i, sim));
            }
        }

        let above_threshold = best.is_some_and(|(_, sim)| sim >= self.threshold);

        if above_threshold {
            let (i, _) = best.unwrap();
            let (c, n) = &mut self.speakers[i];
            *n += 1;
            let k = 1.0 / *n as f32;
            for (cv, ev) in c.iter_mut().zip(emb) {
                *cv += (ev - *cv) * k;
            }
            i + 1
        } else if can_create || self.speakers.is_empty() {
            self.speakers.push((emb.to_vec(), 1));
            self.speakers.len()
        } else {
            let (i, _) = best.unwrap();
            i + 1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn registry_clusters_and_updates_centroids() {
        let mut reg = SpeakerRegistry::new(0.65);
        // two nearly-orthogonal directions
        let a = vec![1.0, 0.0, 0.05];
        let b = vec![0.0, 1.0, 0.05];
        assert_eq!(reg.assign(&a), 1);
        assert_eq!(reg.assign(&b), 2);
        // close to a -> speaker 1 again
        assert_eq!(reg.assign(&[0.95, 0.05, 0.0]), 1);
        assert_eq!(reg.len(), 2);
        // centroid moved toward the new sample
        assert!(reg.centroid(1)[0] < 1.0 && reg.centroid(1)[0] > 0.9);
    }

    #[test]
    fn restore_keeps_numbering() {
        // NOTE: threshold raised from the brief's 0.65 to 0.75. With two
        // orthogonal axis-aligned centroids, the minimum possible max-similarity
        // for *any* unit vector is cos(45 deg) = 0.7071 (equidistant point) --
        // mathematically impossible to fall below a 0.65 threshold, so the
        // literal brief test can never observe a "new speaker" outcome here
        // regardless of implementation. 0.75 preserves the documented intent
        // (45 deg off both axes falls below threshold => new speaker).
        let mut reg = SpeakerRegistry::restore(0.75, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert_eq!(reg.assign(&[0.0, 0.9]), 2);
        assert_eq!(reg.assign(&[0.7, 0.7]), 3); // 45° off both, below threshold
    }

    #[test]
    fn assign_with_policy_denies_new_speaker_without_polluting_centroid() {
        let mut reg = SpeakerRegistry::restore(0.65, vec![vec![1.0, 0.0]]);
        let probe = [0.0, 1.0]; // orthogonal to speaker 1 -> below threshold

        // can_create = false: attributed to nearest existing speaker, centroid untouched.
        assert_eq!(reg.assign_with_policy(&probe, false), 1);
        assert_eq!(reg.centroid(1), &[1.0, 0.0]);

        // can_create = true: same probe now spawns a new speaker.
        assert_eq!(reg.assign_with_policy(&probe, true), 2);
    }

    #[test]
    fn same_voice_embeds_similarly() {
        let p = Path::new("models/nemo_en_titanet_small.onnx");
        assert!(p.exists(), "run scripts/fetch-models.sh tiny");
        let mut e = SpeakerEmbedder::new(p).unwrap();
        let mut r = hound::WavReader::open("tests/fixtures/jfk.wav").unwrap();
        let s: Vec<f32> = r
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let half = s.len() / 2;
        let e1 = e.embed(&s[..half]).unwrap();
        let e2 = e.embed(&s[half..]).unwrap();
        assert!(!e1.is_empty());
        let sim = cosine(&e1, &e2);
        assert!(sim > 0.5, "same speaker halves scored {sim}");
    }
}
