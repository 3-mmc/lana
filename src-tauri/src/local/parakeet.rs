// SPDX-License-Identifier: GPL-3.0-or-later
// NVIDIA's Parakeet TDT 0.6B v3 (CC BY 4.0) through sherpa-onnx: words in 25
// European languages, with punctuation, the language detected by the model.
//
// A TDT transducer predicts each token together with how many frames it
// lasts, so a word's end comes from the model too, not only its start: that
// is the difference from the CTC models, whose words run to the next spike.

use std::path::Path;

use serde_json::{json, Value};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig};

/// The shortest a word may be, and how long the last word of a piece is taken
/// to last when the model gave no duration for its last token.
const MIN_S: f64 = 0.02;
const TAIL_S: f64 = 0.08;

pub struct Parakeet {
    rec: OfflineRecognizer,
}

/// One word: text, start, end.
pub type Word = (String, f64, f64);

impl Parakeet {
    /// The model in `dir`, which holds the four files of its catalogue entry.
    pub fn load(dir: &Path, threads: i32) -> Result<Self, String> {
        let p = |f: &str| Some(dir.join(f).to_string_lossy().into_owned());
        let mut cfg = OfflineRecognizerConfig::default();
        cfg.model_config.transducer = OfflineTransducerModelConfig {
            encoder: p("encoder.int8.onnx"),
            decoder: p("decoder.int8.onnx"),
            joiner: p("joiner.int8.onnx"),
        };
        cfg.model_config.tokens = p("tokens.txt");
        cfg.model_config.model_type = Some("nemo_transducer".into());
        cfg.model_config.num_threads = threads;
        cfg.model_config.provider = Some("cpu".into());
        let rec = OfflineRecognizer::create(&cfg).ok_or("sherpa-onnx could not load Parakeet")?;
        Ok(Parakeet { rec })
    }

    /// The words of one piece of 16 kHz audio, at times `offset` seconds on
    /// and no later than `limit`.
    pub fn words(&self, samples: &[f32], rate: u32, offset: f64, limit: f64) -> Vec<Word> {
        let stream = self.rec.create_stream();
        stream.accept_waveform(rate as i32, samples);
        self.rec.decode(&stream);
        let Some(r) = stream.get_result() else { return Vec::new() };
        let times = r.timestamps.unwrap_or_default();
        let lens = r.durations.unwrap_or_default();
        let toks: Vec<(String, f64, Option<f64>)> = r.tokens.into_iter().enumerate()
            .filter_map(|(i, t)| {
                let at = *times.get(i)? as f64;
                Some((t, offset + at, lens.get(i).map(|d| *d as f64).filter(|d| *d > 0.0)))
            })
            .collect();
        words_of(&toks, limit)
    }
}

/// Word pieces -> words. A leading space (or "▁") starts a word; any other piece (the rest of a
/// word, or punctuation) belongs to the one before. A word ends where its
/// last piece ends by the model's duration, but never after the next word
/// starts; without a duration, it runs to the next word.
pub fn words_of(toks: &[(String, f64, Option<f64>)], limit: f64) -> Vec<Word> {
    // (text, start, end of the last piece if known, start of the last piece)
    let mut acc: Vec<(String, f64, Option<f64>, f64)> = Vec::new();
    for (tok, at, dur) in toks {
        if tok.starts_with('<') && tok.ends_with('>') {
            continue; // <unk>, <blk> and the like
        }
        // sherpa-onnx hands the pieces over decoded: a word's first piece
        // starts with a space. Older exports keep sentencepiece's ▁.
        let marked = tok.starts_with(['▁', ' ']);
        let starts = marked || acc.is_empty();
        let piece = tok.trim_start_matches(['▁', ' ']);
        if starts {
            if piece.is_empty() && marked {
                // A bare ▁ before punctuation or a split token: start the word anyway.
                acc.push((String::new(), *at, dur.map(|d| at + d), *at));
                continue;
            }
            acc.push((piece.to_owned(), *at, dur.map(|d| at + d), *at));
        } else if let Some(w) = acc.last_mut() {
            w.0.push_str(piece);
            w.2 = dur.map(|d| at + d);
            w.3 = *at;
        }
    }
    let acc: Vec<_> = acc.into_iter().filter(|w| !w.0.trim().is_empty()).collect();
    let n = acc.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let (ref text, a, end, last) = acc[k];
        let next = acc.get(k + 1).map(|w| w.1);
        let b = match (end, next) {
            (Some(e), Some(nx)) => e.min(nx),
            (Some(e), None) => e,
            (None, Some(nx)) => nx,
            (None, None) => last + TAIL_S,
        };
        let a = a.min(limit);
        out.push((text.clone(), a, b.min(limit).max(a + MIN_S)));
    }
    out
}

/// Words -> the cue contract, grouped as Omnilingual's are (ctc::word_cues):
/// a cue breaks at a pause of 0.6 s, after twelve words, or after six seconds.
pub fn cues(words: &[Word]) -> Vec<Value> {
    let mut cues: Vec<(f64, f64, Vec<Value>)> = Vec::new();
    for (w, a, b) in words {
        let word = json!({ "text": w, "start": r3(*a), "end": r3(*b), "phones": [], "phone_times": [], "aligned": true });
        match cues.last_mut() {
            Some(c) if a - c.1 < 0.6 && c.2.len() < 12 && b - c.0 <= 6.0 => { c.1 = *b; c.2.push(word); }
            _ => cues.push((*a, *b, vec![word])),
        }
    }
    cues.into_iter().map(|(a, b, ws)| json!({ "start": r3(a), "end": r3(b), "words": ws })).collect()
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(v: &[(&str, f64, f64)]) -> Vec<(String, f64, Option<f64>)> {
        v.iter().map(|(s, a, d)| (s.to_string(), *a, (*d > 0.0).then_some(*d))).collect()
    }

    #[test]
    fn pieces_join_into_words_that_end_by_duration() {
        let w = words_of(&t(&[("▁Der", 0.40, 0.16), (" nörd", 0.64, 0.16), ("lich", 0.80, 0.08), ("e", 0.88, 0.08), (",", 0.96, 0.08), ("▁Teil", 1.60, 0.24)]), 10.0);
        let names: Vec<&str> = w.iter().map(|w| w.0.as_str()).collect();
        assert_eq!(names, ["Der", "nördliche,", "Teil"]);
        assert!((w[0].2 - 0.56).abs() < 1e-9, "Der ends by its duration, not at the next word");
        assert!((w[1].2 - 1.04).abs() < 1e-9);
        assert!((w[2].2 - 1.84).abs() < 1e-9);
    }

    #[test]
    fn without_durations_a_word_runs_to_the_next() {
        let w = words_of(&t(&[("▁a", 0.1, 0.0), ("▁b", 0.5, 0.0)]), 0.55);
        assert_eq!((w[0].1, w[0].2), (0.1, 0.5));
        assert_eq!(w[1].2, 0.55, "held inside the piece");
    }
}
