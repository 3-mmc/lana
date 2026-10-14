// SPDX-License-Identifier: GPL-3.0-or-later
// CTC recognisers run through sherpa-onnx: ZIPA (Zhu et al. 2025) for IPA
// phones, and Meta's Omnilingual ASR for words in some 1600 languages. Both
// emit one symbol per frame with its time, which is what makes their
// boundaries come from the model rather than from a separate aligner.

use std::path::Path;

use serde_json::{json, Value};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    /// ZIPA: IPA phones, a zipformer CTC model.
    Zipa,
    /// Omnilingual ASR: characters with an explicit space, from raw audio.
    Omnilingual,
}

impl Engine {
    /// What to multiply sherpa-onnx's timestamps by. It assumes a zipformer
    /// CTC model emits one frame per 40 ms (10 ms features, subsampled by 4);
    /// ZIPA emits one per 20 ms, so every time it reports is exactly twice the
    /// real one. Measured on the FLEURS demo clip: the first phone at 1.52 s
    /// from sherpa-onnx, 0.764 s from ZIPA's reference decoder over the same
    /// frames. For Omnilingual it assumes 20 ms, which is right.
    fn time_scale(self) -> f64 {
        match self {
            Engine::Zipa => 0.5,
            Engine::Omnilingual => 1.0,
        }
    }
}
/// Audio is recognised in pieces of at most this long (Whisper too: see
/// Local::run). The encoder attends
/// over the whole input, so a long recording in one pass costs memory that
/// grows with its square; pieces also let Stop and the progress bar work.
const PIECE_S: f64 = 30.0;
/// A pause this long between two phones starts a new group in the response.
const PAUSE_S: f64 = 0.35;
/// How long the last phone of a group is taken to last.
const TAIL_S: f64 = 0.08;

pub struct Ctc {
    rec: OfflineRecognizer,
    engine: Engine,
}

impl Ctc {
    pub fn load(engine: Engine, model: &Path, tokens: &Path, threads: i32) -> Result<Self, String> {
        let mut cfg = OfflineRecognizerConfig::default();
        let model = Some(model.to_string_lossy().into_owned());
        match engine {
            Engine::Zipa => cfg.model_config.zipformer_ctc.model = model,
            Engine::Omnilingual => cfg.model_config.omnilingual.model = model,
        }
        cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        cfg.model_config.num_threads = threads;
        cfg.model_config.provider = Some("cpu".into());
        let rec = OfflineRecognizer::create(&cfg).ok_or("sherpa-onnx could not load the model")?;
        Ok(Ctc { rec, engine })
    }

    /// Symbols with their times for 16 kHz mono audio: phones for ZIPA,
    /// characters for Omnilingual, where " " separates words. `on_piece` gets
    /// the seconds done and the symbols of the piece just finished, so a caller
    /// can show them before the whole recording is read; it returns false to stop.
    pub fn recognise(&self, samples: &[f32], rate: u32, mut on_piece: impl FnMut(f64, &[(String, f64)]) -> bool) -> Result<Vec<(String, f64)>, String> {
        let mut out = Vec::new();
        let mut at = 0usize;
        while at < samples.len() {
            let piece_from = out.len();
            let end = cut_point(samples, at, rate);
            let stream = self.rec.create_stream();
            stream.accept_waveform(rate as i32, &samples[at..end]);
            self.rec.decode(&stream);
            let offset = at as f64 / rate as f64;
            if let Some(r) = stream.get_result() {
                let times = r.timestamps.unwrap_or_default();
                let scale = self.engine.time_scale();
                for (tok, t) in r.tokens.iter().zip(times) {
                    let at = offset + t as f64 * scale;
                    if self.engine == Engine::Omnilingual {
                        // Characters, and a space (or "|") between words.
                        let tok = if tok.trim().is_empty() || tok == "|" { " " } else { tok.as_str() };
                        if !tok.starts_with('<') {
                            out.push((tok.to_owned(), at));
                        }
                        continue;
                    }
                    // ▁ is ZIPA's word marker; it emits one at the start of a
                    // piece and nothing a phone tier can use.
                    let tok = tok.trim().trim_start_matches('▁');
                    if tok.is_empty() || tok.starts_with('<') {
                        continue;
                    }
                    out.push((tok.to_owned(), at));
                }
                // A word cannot continue across a piece boundary: close the
                // last one just after its last character, not at the cut,
                // which may be well into a pause.
                if self.engine == Engine::Omnilingual {
                    if let Some(&(_, last)) = out.last() {
                        out.push((" ".into(), last + TAIL_S));
                    }
                }
            }
            at = end;
            if !on_piece(at as f64 / rate as f64, &out[piece_from..]) {
                return Err("stopped".into());
            }
        }
        Ok(out)
    }
}

/// Where the piece starting at `at` should end: at most PIECE_S later, at the
/// quietest 20 ms in its last five seconds, so a cut lands in a pause rather
/// than through a phone.
pub(crate) fn cut_point(x: &[f32], at: usize, rate: u32) -> usize {
    let piece = (PIECE_S * rate as f64) as usize;
    if x.len() - at <= piece {
        return x.len();
    }
    let hop = (0.02 * rate as f64) as usize;
    let from = at + piece - 5 * rate as usize;
    let mut best = (f32::MAX, at + piece);
    let mut i = from;
    while i + hop <= at + piece {
        let e: f32 = x[i..i + hop].iter().map(|v| v * v).sum();
        if e < best.0 {
            best = (e, i + hop / 2);
        }
        i += hop;
    }
    best.1
}

/// Characters -> words -> the cue contract. A word runs from its first
/// character to the space after it, or to the next word if that comes first;
/// cues break at a pause of 0.6 s, after twelve words, or after six seconds.
pub fn word_cues(chars: &[(String, f64)], duration: f64) -> Vec<Value> {
    let mut words: Vec<(String, f64, f64)> = Vec::new();
    let mut cur: Option<(String, f64, f64)> = None;
    for (c, t) in chars {
        if c == " " {
            if let Some((w, a, last)) = cur.take() {
                words.push((w, a, t.max(last + 0.02)));
            }
        } else {
            match cur.as_mut() {
                Some(w) => { w.0.push_str(c); w.2 = *t; }
                None => cur = Some((c.clone(), *t, *t)),
            }
        }
    }
    if let Some((w, a, last)) = cur {
        words.push((w, a, (last + TAIL_S).min(duration)));
    }
    // A word may not run past the start of the next one.
    for i in 0..words.len().saturating_sub(1) {
        let next = words[i + 1].1;
        if words[i].2 > next { words[i].2 = next.max(words[i].1 + 0.02); }
    }
    let mut cues: Vec<(f64, f64, Vec<Value>)> = Vec::new();
    for (w, a, b) in words {
        let word = json!({ "text": w, "start": r3(a), "end": r3(b), "phones": [], "phone_times": [], "aligned": true });
        match cues.last_mut() {
            Some(c) if a - c.1 < 0.6 && c.2.len() < 12 && b - c.0 <= 6.0 => { c.1 = b; c.2.push(word); }
            _ => cues.push((a, b, vec![word])),
        }
    }
    cues.into_iter().map(|(a, b, ws)| json!({ "start": r3(a), "end": r3(b), "words": ws })).collect()
}

/// Phones -> the cue contract. A group of phones between pauses is one cue
/// holding one word with no text: the phones are real, a word is not claimed.
/// Each phone runs to the next one's start; the last of a group gets TAIL_S.
pub fn cues(phones: &[(String, f64)], duration: f64) -> Vec<Value> {
    let mut groups: Vec<Vec<&(String, f64)>> = Vec::new();
    for p in phones {
        match groups.last_mut() {
            Some(g) if p.1 - g.last().unwrap().1 < PAUSE_S => g.push(p),
            _ => groups.push(vec![p]),
        }
    }
    groups
        .iter()
        .map(|g| {
            let start = g[0].1;
            let end = (g.last().unwrap().1 + TAIL_S).min(duration).max(start + 0.02);
            let word = json!({
                "text": "", "start": r3(start), "end": r3(end),
                "phones": g.iter().map(|p| p.0.clone()).collect::<Vec<_>>(),
                "phone_times": g.iter().map(|p| r3(p.1)).collect::<Vec<_>>(),
                "aligned": true,
            });
            json!({ "start": r3(start), "end": r3(end), "words": [word] })
        })
        .collect()
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_split_at_pauses_and_end_after_the_last_phone() {
        let p: Vec<(String, f64)> = [("a", 0.10), ("n", 0.20), ("t", 0.30), ("i", 1.00), ("s", 1.10)]
            .iter().map(|(s, t)| (s.to_string(), *t)).collect();
        let c = cues(&p, 2.0);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["words"][0]["phones"], json!(["a", "n", "t"]));
        assert_eq!(c[0]["end"], json!(0.38));
        assert_eq!(c[1]["start"], json!(1.0));
    }

    #[test]
    fn characters_become_timed_words() {
        let c: Vec<(String, f64)> = [("d", 0.64), ("e", 0.68), ("r", 0.72), (" ", 0.76), ("n", 0.80), ("o", 0.86), (" ", 1.20)]
            .iter().map(|(s, t)| (s.to_string(), *t)).collect();
        let cues = word_cues(&c, 2.0);
        let w = &cues[0]["words"];
        assert_eq!(w[0]["text"], json!("der"));
        assert_eq!((w[0]["start"].clone(), w[0]["end"].clone()), (json!(0.64), json!(0.76)));
        assert_eq!(w[1]["text"], json!("no"));
        assert_eq!(w[1]["end"], json!(1.2));
    }

    #[test]
    fn short_audio_is_one_piece_and_long_audio_is_cut_in_the_quietest_place() {
        let rate = 16_000;
        assert_eq!(cut_point(&vec![0.1; rate * 10], 0, rate as u32), rate * 10);
        let mut x = vec![0.5f32; rate * 40];
        for v in &mut x[rate * 27..rate * 27 + 800] { *v = 0.0; }
        let cut = cut_point(&x, 0, rate as u32);
        assert!((rate * 27..rate * 27 + 800).contains(&cut), "cut at {cut}");
    }
}
