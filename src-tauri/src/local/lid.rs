// SPDX-License-Identifier: GPL-3.0-or-later
// Which language is spoken where: MMS-LID (Meta, 2023, CC BY-NC 4.0) over
// short windows, for a recording the linguist has marked as codeswitching.
// The language tier it fills then decides the language each word model reads
// a stretch in.
//
// MMS-LID is wav2vec2 (1B) with a mean pool and one linear layer on top, and
// the first idea was to run the encoder once over a long piece and pool its
// frames per window, the head being linear. It does not work: the encoder
// attends across its whole input, so every frame already speaks for the
// piece. Pooled over 3 s windows of a 27 s Georgian-then-German piece, every
// window read Georgian (58% of the time right, no switch found), where the
// classifier run on each window's audio alone found the switch within 0.4 s
// (98.6%). So each window is classified on its own, as here.
//
// Windows are WINDOW_S long, one every STEP_S, centred; a window that holds
// less than MIN_SPEECH_S of speech by Silero VAD is not classified. A Viterbi
// pass over each run of windows then picks the languages, paying SWITCH for
// each change, which keeps a lone window from flipping the language. When
// the linguist names the languages that occur, the classifier chooses among
// those alone (its logits renormalised over them). See ROADMAP.md and
// .agent-memory/lana-mms-lid.md for the measurements behind the numbers.

use std::{ops::Range, path::Path, sync::Mutex};

use ort::{session::Session, value::Tensor};
use serde_json::{json, Value};

pub const RATE: usize = 16_000;
const WINDOW_S: f64 = 3.0;
const STEP_S: f64 = 1.0;
const MIN_SPEECH_S: f64 = 0.5;
/// The log-probability a change of language costs.
const SWITCH: f32 = 3.0;

pub struct Lid {
    session: Mutex<Session>,
    labels: Vec<String>,
}

/// One stretch of one language.
#[derive(Clone, Debug)]
pub struct Stretch {
    pub start: f64,
    pub end: f64,
    pub lang: String,
    /// The mean share the classifier gave this language over the stretch's
    /// windows, and the runner-ups' (at least 0.05, at most three).
    pub conf: f32,
    pub alts: Vec<(String, f32)>,
}

/// Codes that name the same language differently in Lana's table (ISO 639-3
/// macrolanguages, mostly) and in MMS-LID's labels: [Lana's, MMS-LID's].
const ALIASES: &[(&str, &str)] = &[
    ("zho", "cmn"), ("fas", "pes"), ("msa", "zlm"), ("swa", "swh"), ("nep", "npi"), ("est", "ekk"),
    ("lav", "lvs"), ("uzb", "uzn"), ("mon", "khk"), ("aze", "azj"), ("sqi", "als"), ("ori", "ory"),
    ("ara", "arb"), ("kur", "kmr"), ("pus", "pbt"), ("grn", "gug"), ("que", "quy"), ("yid", "ydd"),
];

impl Lid {
    pub fn load(model: &Path, config: &Path, threads: i32) -> Result<Self, String> {
        let cfg: Value = serde_json::from_slice(&std::fs::read(config).map_err(|e| format!("{}: {e}", config.display()))?)
            .map_err(|e| format!("config.json: {e}"))?;
        let map = cfg["id2label"].as_object().ok_or("config.json has no labels")?;
        let mut labels = vec![String::new(); map.len()];
        for (k, v) in map {
            let i: usize = k.parse().map_err(|_| "config.json: a label id is not a number")?;
            *labels.get_mut(i).ok_or("config.json: label ids are not 0..n")? = v.as_str().unwrap_or("").to_owned();
        }
        let session = Session::builder(super::frames::environment()?)
            .and_then(|b| b.with_intra_threads(threads.max(1) as usize).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(model))
            .map_err(|e| format!("could not load {}: {e}", model.display()))?;
        Ok(Lid { session: Mutex::new(session), labels })
    }

    /// The label indices for a list of Lana's language codes, and the codes
    /// MMS-LID does not know. An empty list means every label.
    pub fn candidates(&self, codes: &[String]) -> (Vec<usize>, Vec<String>) {
        let (mut idx, mut unknown) = (Vec::new(), Vec::new());
        for c in codes {
            let c = c.trim().to_lowercase();
            let alias = ALIASES.iter().find(|(a, b)| *a == c || *b == c).map(|(a, b)| if *a == c { *b } else { *a });
            match self.labels.iter().position(|l| *l == c).or_else(|| alias.and_then(|a| self.labels.iter().position(|l| l == a))) {
                Some(i) if !idx.contains(&i) => idx.push(i),
                Some(_) => {}
                None => unknown.push(c),
            }
        }
        if idx.is_empty() {
            idx = (0..self.labels.len()).collect();
        }
        (idx, unknown)
    }

    /// Logits for one window of 16 kHz audio, normalised as MMS was trained.
    fn logits(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let err = |e: ort::Error| format!("the language model failed: {e}");
        let t = Tensor::from_array(([1usize, x.len()], normalised(x))).map_err(err)?;
        let mut session = self.session.lock().unwrap();
        let out = session.run(ort::inputs!["input_values" => t]).map_err(err)?;
        let (_, lg) = out["logits"].try_extract_tensor::<f32>().map_err(err)?;
        Ok(lg.to_vec())
    }

    /// The recording's language stretches. `speech` is where Silero VAD heard
    /// speech (None: everywhere). `on_progress` gets seconds done and returns
    /// false to stop.
    pub fn stretches(&self, x: &[f32], speech: Option<&[Range<usize>]>, cand: &[usize],
                     mut on_progress: impl FnMut(f64) -> bool) -> Result<Vec<Stretch>, String> {
        let dur = x.len() as f64 / RATE as f64;
        let n = (dur / STEP_S).floor() as usize + 1;
        // Per window: log-probabilities over the candidates, or None (no speech).
        let mut lp: Vec<Option<Vec<f32>>> = Vec::with_capacity(n);
        for k in 0..n {
            let c = k as f64 * STEP_S;
            let (a, b) = (((c - WINDOW_S / 2.0).max(0.0) * RATE as f64) as usize,
                          (((c + WINDOW_S / 2.0) * RATE as f64) as usize).min(x.len()));
            let heard = speech.map_or((b - a) as f64, |s| s.iter().map(|r| r.end.min(b).saturating_sub(r.start.max(a)) as f64).sum()) / RATE as f64;
            if b - a < RATE / 2 || heard < MIN_SPEECH_S {
                lp.push(None);
            } else {
                let lg = self.logits(&x[a..b])?;
                let mut row: Vec<f32> = cand.iter().map(|&i| lg[i]).collect();
                log_softmax(&mut row);
                lp.push(Some(row));
            }
            if !on_progress(c.min(dur)) {
                return Err("stopped".into());
            }
        }
        let mut out = Vec::new();
        let mut k = 0;
        while k < n {
            if lp[k].is_none() { k += 1; continue; }
            let first = k;
            while k < n && lp[k].is_some() { k += 1; }
            let run: Vec<&Vec<f32>> = lp[first..k].iter().map(|r| r.as_ref().unwrap()).collect();
            let path = viterbi(&run, SWITCH);
            let at = |w: usize| (w as f64 * STEP_S - STEP_S / 2.0).clamp(0.0, dur);
            let mut s = 0;
            while s < path.len() {
                let mut e = s;
                while e < path.len() && path[e] == path[s] { e += 1; }
                // Mean share of every candidate over the stretch's windows.
                let mut share = vec![0f32; cand.len()];
                for row in &run[s..e] {
                    for (j, v) in row.iter().enumerate() { share[j] += v.exp() / (e - s) as f32; }
                }
                let mut alts: Vec<(String, f32)> = share.iter().enumerate()
                    .filter(|(j, v)| *j != path[s] && **v >= 0.05)
                    .map(|(j, v)| (self.labels[cand[j]].clone(), *v)).collect();
                alts.sort_by(|a, b| b.1.total_cmp(&a.1));
                alts.truncate(3);
                out.push(Stretch { start: at(first + s), end: at(first + e).max(at(first + s)),
                                   lang: self.labels[cand[path[s]]].clone(), conf: share[path[s]], alts });
                s = e;
            }
            if let Some(last) = out.last_mut() {
                last.end = ((k - 1) as f64 * STEP_S + STEP_S / 2.0).min(dur);
            }
        }
        Ok(merge(out))
    }
}

/// Neighbouring stretches of one language, split only by a window with no
/// speech, are one stretch; the pause stays inside it.
fn merge(v: Vec<Stretch>) -> Vec<Stretch> {
    let mut out: Vec<Stretch> = Vec::new();
    for s in v {
        if let Some(last) = out.last_mut().filter(|l| l.lang == s.lang && s.start - l.end < 2.0 * STEP_S) {
            let (a, b) = (last.end - last.start, s.end - s.start);
            last.conf = (last.conf * a as f32 + s.conf * b as f32) / (a + b).max(1e-6) as f32;
            last.end = s.end;
            continue;
        }
        out.push(s);
    }
    out
}

/// The likeliest sequence of candidates through the windows, a change
/// costing `switch`.
fn viterbi(lp: &[&Vec<f32>], switch: f32) -> Vec<usize> {
    let k = lp[0].len();
    let mut score = lp[0].clone();
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(lp.len());
    for row in &lp[1..] {
        let best = (0..k).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap();
        let mv = score[best] - switch;
        let mut bp = vec![0; k];
        let mut next = vec![0f32; k];
        for j in 0..k {
            if score[j] >= mv { bp[j] = j; next[j] = score[j] + row[j]; } else { bp[j] = best; next[j] = mv + row[j]; }
        }
        back.push(bp);
        score = next;
    }
    let mut j = (0..k).max_by(|&a, &b| score[a].total_cmp(&score[b])).unwrap();
    let mut path = vec![j];
    for bp in back.iter().rev() {
        j = bp[j];
        path.push(j);
    }
    path.reverse();
    path
}

fn normalised(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|&v| ((v as f64 - mean) * inv) as f32).collect()
}

fn log_softmax(row: &mut [f32]) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    for v in row {
        *v -= log;
    }
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Stretches as cues: each carries its language as `lang` and as its one
/// word, with the confidence and runner-ups the doubt and triage UI read.
pub fn cues(v: &[Stretch]) -> Vec<Value> {
    v.iter().map(|s| {
        let alts: Vec<Value> = s.alts.iter().map(|(l, p)| json!([l, round3(*p as f64)])).collect();
        json!({ "start": round3(s.start), "end": round3(s.end), "lang": s.lang,
                "words": [{ "text": s.lang, "start": round3(s.start), "end": round3(s.end),
                            "conf": round3(s.conf as f64), "alts": alts }] })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lone_window_does_not_switch_the_language() {
        let a = vec![-0.1f32, -2.4];
        let b = vec![-2.0f32, -0.15];
        let rows = [&a, &a, &b, &a, &a, &b, &b, &b, &b];
        assert_eq!(viterbi(&rows, SWITCH), vec![0, 0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(viterbi(&rows, 0.0), vec![0, 0, 1, 0, 0, 1, 1, 1, 1]);
    }

    /// Stretches for recordings, as JSON lines, for measuring:
    /// LID_DIR=<folder with model.int8.onnx, config.json> LID_LANGS=kat,deu
    /// cargo test --release -- --ignored lid_files -- a.wav b.wav (LID_WAVS)
    #[test]
    #[ignore]
    fn lid_files() {
        let dir = std::path::PathBuf::from(std::env::var("LID_DIR").unwrap());
        let vad = std::env::var("LID_VAD").ok();
        if let Some(v) = &vad { crate::local::vad::use_model(v.into()); }
        let t = std::time::Instant::now();
        let lid = Lid::load(&dir.join("model.int8.onnx"), &dir.join("config.json"), 7).unwrap();
        eprintln!("load {:.1} s", t.elapsed().as_secs_f64());
        let codes: Vec<String> = std::env::var("LID_LANGS").unwrap_or_default().split(',').filter(|c| !c.is_empty()).map(String::from).collect();
        let (cand, unknown) = lid.candidates(&codes);
        for wav in std::env::var("LID_WAVS").unwrap().split_whitespace() {
            let x = crate::local::audio::decode_16k_mono(std::fs::read(wav).unwrap(), wav).unwrap();
            let t = std::time::Instant::now();
            let speech = crate::local::vad::speech(&x, RATE as u32);
            let s = lid.stretches(&x, speech.as_deref(), &cand, |_| true).unwrap();
            let secs = t.elapsed().as_secs_f64();
            println!("{}", json!({ "wav": wav, "audio_s": x.len() as f64 / RATE as f64, "secs": secs, "unknown": unknown, "cues": cues(&s) }));
        }
    }

    #[test]
    fn stretches_split_by_a_pause_merge() {
        let s = |a: f64, b: f64, l: &str| Stretch { start: a, end: b, lang: l.into(), conf: 0.9, alts: vec![] };
        let m = merge(vec![s(0.0, 4.5, "kat"), s(5.5, 9.0, "kat"), s(9.0, 12.0, "rus")]);
        assert_eq!(m.len(), 2);
        assert_eq!((m[0].start, m[0].end, m[1].lang.as_str()), (0.0, 9.0, "rus"));
    }
}
