// SPDX-License-Identifier: GPL-3.0-or-later
// NVIDIA's Parakeet TDT 0.6B v3 (CC BY 4.0), in sherpa-onnx's export: words
// in 25 European languages, with punctuation, the language detected by the
// model. Run here as sherpa-onnx 1.13.8 ran it (offline-recognizer-
// transducer-nemo-impl.h, greedy TDT search), on Lana's own runtime.
//
// A TDT transducer predicts each token together with how many frames it
// lasts, so a word's end comes from the model too, not only its start: that
// is the difference from the CTC models, whose words run to the next spike.

use std::{path::Path, sync::Mutex};

use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};

use super::{frames::Fbank, runtime::{self, Role}};

/// The shortest a word may be, and how long the last word of a piece is taken
/// to last when the model gave no duration for its last token.
const MIN_S: f64 = 0.02;
const TAIL_S: f64 = 0.08;

pub struct Parakeet {
    encoder: Mutex<runtime::Model>,
    net: Mutex<(Session, Session)>,
    tokens: Vec<String>,
    fbank: Fbank,
}

/// One word: text, start, end.
pub type Word = (String, f64, f64);

/// The encoder's output frames are 8 feature frames of 10 ms apart.
const FRAME_S: f64 = 0.08;
/// The prediction network: 2 LSTM layers of 640.
const LAYERS: usize = 2;
const HIDDEN: usize = 640;
/// At most this many tokens on one frame before the search moves on.
const MAX_TOKENS_PER_FRAME: usize = 5;

impl Parakeet {
    /// The model whose encoder is `encoder` (int8, or its float variant for
    /// an accelerator), with the other files of its catalogue entry beside it.
    pub fn load(encoder: &Path, threads: i32) -> Result<Self, String> {
        let dir = encoder.parent().ok_or("no model folder")?;
        let text = std::fs::read_to_string(dir.join("tokens.txt")).map_err(|e| format!("tokens.txt: {e}"))?;
        let mut tokens = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            let (tok, id) = line.rsplit_once(' ').ok_or("tokens.txt: a line without an id")?;
            if id.parse::<usize>().ok() != Some(tokens.len()) {
                return Err("tokens.txt is not in order".into());
            }
            // sentencepiece marks a word's start with ▁; sherpa-onnx hands
            // it over as a space, and so does Lana.
            tokens.push(tok.replace('▁', " "));
        }
        if tokens.last().map(String::as_str) != Some("<blk>") {
            return Err("tokens.txt does not end with the blank <blk>".into());
        }
        let encoder = runtime::Model::load(encoder, Role::Heavy, threads)?;
        let decoder = runtime::session(&dir.join("decoder.int8.onnx"), Role::Light, 1)?;
        let joiner = runtime::session(&dir.join("joiner.int8.onnx"), Role::Light, 1)?;
        Ok(Parakeet { encoder: Mutex::new(encoder), net: Mutex::new((decoder, joiner)), tokens, fbank: Fbank::librosa(128) })
    }

    /// The words of one piece of 16 kHz audio, at times `offset` seconds on
    /// and no later than `limit`.
    pub fn words(&self, samples: &[f32], _rate: u32, offset: f64, limit: f64) -> Vec<Word> {
        match self.tokens_of(samples) {
            Ok(toks) => {
                let toks: Vec<(String, f64, Option<f64>)> = toks.into_iter()
                    .map(|(t, at, dur)| (t, offset + at, Some(dur).filter(|d| *d > 0.0)))
                    .collect();
                words_of(&toks, limit)
            }
            Err(e) => {
                eprintln!("lana: Parakeet failed: {e}");
                Vec::new()
            }
        }
    }

    /// Tokens with their start and duration in seconds.
    fn tokens_of(&self, samples: &[f32]) -> Result<Vec<(String, f64, f64)>, String> {
        let err = |e: ort::Error| format!("Parakeet: {e}");
        let (frames, mut feats) = self.fbank.compute(samples);
        if frames == 0 {
            return Ok(Vec::new());
        }
        let bins = self.fbank.bins;
        normalise_per_feature(&mut feats, frames, bins);
        // (time, feature) -> (feature, time), as the encoder takes it.
        let mut x = vec![0f32; frames * bins];
        for t in 0..frames {
            for f in 0..bins {
                x[f * frames + t] = feats[t * bins + f];
            }
        }
        let (enc, dim, len) = self.encoder.lock().unwrap().with(|encoder| {
            let x = Tensor::from_array(([1usize, bins, frames], x.clone())).map_err(err)?;
            let lens = Tensor::from_array(([1usize], vec![frames as i64])).map_err(err)?;
            let out = encoder.run(ort::inputs![x, lens]).map_err(err)?;
            let (shape, e) = out[0].try_extract_tensor::<f32>().map_err(err)?;
            let (_, l) = out[1].try_extract_tensor::<i64>().map_err(err)?;
            let (dim, t) = (shape[1] as usize, shape[2] as usize);
            // (dim, time) -> (time, dim): one row per output frame.
            let mut rows = vec![0f32; dim * t];
            for d in 0..dim {
                for i in 0..t {
                    rows[i * dim + d] = e[d * t + i];
                }
            }
            Ok((rows, dim, (l[0].max(0) as usize).min(t)))
        })?;
        let mut net = self.net.lock().unwrap();
        let (decoder, joiner) = &mut *net;
        let blank = self.tokens.len() - 1;
        let vocab = self.tokens.len();
        let mut state = (vec![0f32; LAYERS * HIDDEN], vec![0f32; LAYERS * HIDDEN]);
        let mut pred = predict(decoder, blank, &mut state)?;
        let mut out = Vec::new();
        let (mut t, mut on_frame) = (0usize, 0usize);
        while t < len {
            let e = Tensor::from_array(([1usize, dim, 1], enc[t * dim..(t + 1) * dim].to_vec())).map_err(err)?;
            let p = Tensor::from_array(([1usize, HIDDEN, 1], pred.clone())).map_err(err)?;
            let o = joiner.run(ort::inputs![e, p]).map_err(err)?;
            let (_, logits) = o[0].try_extract_tensor::<f32>().map_err(err)?;
            let (tok, dur) = (&logits[..vocab], &logits[vocab..]);
            let y = argmax(tok);
            let mut skip = argmax(dur);
            if y != blank {
                out.push((self.tokens[y].clone(), t as f64 * FRAME_S, skip as f64 * FRAME_S));
                pred = predict(decoder, y, &mut state)?;
                on_frame += 1;
            }
            if skip > 0 {
                on_frame = 0;
            }
            if on_frame >= MAX_TOKENS_PER_FRAME || (y == blank && skip == 0) {
                on_frame = 0;
                skip = 1;
            }
            t += skip;
        }
        Ok(out)
    }
}

/// The prediction network after `token`: its output, with the LSTM state
/// carried forward.
fn predict(decoder: &mut Session, token: usize, state: &mut (Vec<f32>, Vec<f32>)) -> Result<Vec<f32>, String> {
    let err = |e: ort::Error| format!("Parakeet: {e}");
    let y = Tensor::from_array(([1usize, 1], vec![token as i32])).map_err(err)?;
    let n = Tensor::from_array(([1usize], vec![1i32])).map_err(err)?;
    let h = Tensor::from_array(([LAYERS, 1, HIDDEN], std::mem::take(&mut state.0))).map_err(err)?;
    let c = Tensor::from_array(([LAYERS, 1, HIDDEN], std::mem::take(&mut state.1))).map_err(err)?;
    let o = decoder.run(ort::inputs![y, n, h, c]).map_err(err)?;
    state.0 = o[2].try_extract_tensor::<f32>().map_err(err)?.1.to_vec();
    state.1 = o[3].try_extract_tensor::<f32>().map_err(err)?.1.to_vec();
    Ok(o[0].try_extract_tensor::<f32>().map_err(err)?.1.to_vec())
}

/// The first index of the largest value.
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b })
}

/// NeMo's per_feature normalisation: each band to zero mean and unit
/// deviation over the piece (the deviation from centred values, plus 1e-5).
fn normalise_per_feature(x: &mut [f32], frames: usize, bins: usize) {
    for f in 0..bins {
        let mean = (0..frames).map(|t| x[t * bins + f]).sum::<f32>() / frames as f32;
        let var = (0..frames).map(|t| (x[t * bins + f] - mean).powi(2)).sum::<f32>() / frames as f32;
        let inv = 1.0 / (var.sqrt() + 1e-5);
        for t in 0..frames {
            x[t * bins + f] = (x[t * bins + f] - mean) * inv;
        }
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
