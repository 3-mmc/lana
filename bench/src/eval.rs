// SPDX-License-Identifier: GPL-3.0-or-later
// lana-bench --eval <dir>: how close Lana's phone and word boundaries come to
// a reference, on DoReCo recordings (Seifart, Paschen & Stave 2024), which
// carry phone-aligned TextGrids for some fifty mostly endangered languages.
//
// DoReCo's word boundaries were checked and corrected by hand; its phone
// boundaries inside each word are WebMAUS's language-independent forced
// alignment (Paschen et al. 2020, sec. 4.4). So the phone figures compare one
// aligner with another, and the word-onset figures are the ones with a human
// behind the reference.
//
// Three measures, for every model given:
// - free: the model reads the recording as the app does (Local::transcribe),
//   and its onsets are matched one to one with the reference's within 20 ms,
//   ignoring the labels: precision, recall, F1 and R-value.
// - forced: the reference phones, as IPA, go to Lana's re-timing
//   (Local::align, behind /api/local/align) one stretch at a time, and each
//   placed onset is compared with the reference's.
// - per: phone error rate of the free reading against the reference, per
//   stretch; only for the phone models.
//
// A stretch is a run of ordinary words inside one of DoReCo's annotation
// units (its ref tier). Labelled words (<<fm>...>, <<fs>>, <<ui>>: foreign
// material, false starts, unclear speech) have no phones and end a run, and
// a run that overlaps another speaker's is left out.
//
// The reference is X-SAMPA; data/xsampa.tsv turns it into IPA. Symbols it
// does not list, and IPA the model has no symbol for, are counted and
// reported, never guessed.

use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
    time::Instant,
};

use serde_json::{json, Value};

use crate::local::{audio, models::{self, Kind}, Local, Models, Request};

/// Within this, a boundary counts as found (Kreuk et al. 2020 and most phone
/// segmentation work since use the same).
const TOL: f64 = 0.02;
const TOL_WIDE: f64 = 0.05;
/// A predicted onset this far from every reference one is not an offset but
/// a miss, and is left out of the median offset.
const NEAR: f64 = 0.1;

const DEFAULT_MODELS: &[&str] = &["zipa-small", "zipa-large", "omnilingual-300m"];

// ------------------------------------------------------------ reference

#[derive(Clone, Debug)]
struct Iv {
    a: f64,
    b: f64,
    text: String,
}

/// A long-format TextGrid's interval tiers, by name. DoReCo writes this
/// format; Praat's short format is not read.
fn read_textgrid(text: &str) -> Result<Vec<(String, Vec<Iv>)>, String> {
    let mut tiers: Vec<(String, Vec<Iv>)> = Vec::new();
    let (mut a, mut b) = (None, None);
    let mut in_intervals = false;
    let mut lines = text.lines().map(str::trim);
    if !lines.next().is_some_and(|l| l.contains("ooTextFile")) {
        return Err("not a TextGrid".into());
    }
    let value = |l: &str| l.split_once('=').map(|(_, v)| v.trim().to_string());
    let unquote = |v: &str| {
        let v = v.strip_prefix('"').unwrap_or(v);
        v.strip_suffix('"').unwrap_or(v).replace("\"\"", "\"")
    };
    while let Some(l) = lines.next() {
        if l.starts_with("name =") {
            tiers.push((unquote(&value(l).unwrap_or_default()), Vec::new()));
            in_intervals = false;
        } else if l.starts_with("intervals [") {
            in_intervals = true;
            (a, b) = (None, None);
        } else if in_intervals && l.starts_with("xmin =") {
            a = value(l).and_then(|v| v.parse::<f64>().ok());
        } else if in_intervals && l.starts_with("xmax =") {
            b = value(l).and_then(|v| v.parse::<f64>().ok());
        } else if in_intervals && l.starts_with("text =") {
            let mut v = value(l).unwrap_or_default();
            // A text may run over several lines. Quotes inside it are
            // doubled, so it is complete once its quotes are even.
            while v.matches('"').count() % 2 == 1 {
                match lines.next() {
                    Some(more) => { v.push('\n'); v.push_str(more); }
                    None => break,
                }
            }
            if let (Some(a), Some(b), Some(t)) = (a, b, tiers.last_mut()) {
                t.1.push(Iv { a, b, text: unquote(&v) });
            }
        } else if l.starts_with("points [") {
            in_intervals = false;
        }
    }
    if tiers.is_empty() {
        return Err("no tiers (only long-format TextGrids are read)".into());
    }
    Ok(tiers)
}

/// X-SAMPA -> IPA pairs, longest X-SAMPA first.
fn xsampa() -> &'static [(String, String)] {
    static T: std::sync::OnceLock<Vec<(String, String)>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t: Vec<(String, String)> = include_str!("../data/xsampa.tsv")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| l.split_once('\t').map(|(x, i)| (x.to_string(), i.to_string())))
            .collect();
        t.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        t
    })
}

/// One DoReCo phone as IPA, and the X-SAMPA characters no entry covers.
fn to_ipa(x: &str) -> (String, Vec<char>) {
    let (mut out, mut bad) = (String::new(), Vec::new());
    let mut rest = x;
    while let Some(c) = rest.chars().next() {
        match xsampa().iter().find(|(k, _)| rest.starts_with(k.as_str())) {
            Some((k, v)) => { out.push_str(v); rest = &rest[k.len()..]; }
            None => { bad.push(c); rest = &rest[c.len_utf8()..]; }
        }
    }
    (out, bad)
}

/// A mark that belongs to the letter before it (ː ʼ ʰ ʷ ˤ, a combining tilde):
/// spacing modifier letters and combining diacritics.
fn is_mark(c: char) -> bool {
    matches!(c as u32, 0x02B0..=0x02FF | 0x0300..=0x036F | 0x1DC0..=0x1DFF)
}

/// IPA as phone units: a letter with the marks after it. Both sides of the
/// error rate are cut this way, so tʃ and the diphthong eɪ are two units in
/// the reference as in what ZIPA writes.
fn units(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for c in s.chars() {
        match out.last_mut() {
            Some(u) if is_mark(c) => u.push(c),
            _ => out.push(c.to_string()),
        }
    }
    out
}

/// A stretch of one speaker's words, with their phones.
struct Au {
    a: f64,
    b: f64,
    /// Onset and IPA of each phone.
    phones: Vec<(f64, String)>,
    /// Onset and text of each word.
    words: Vec<(f64, String)>,
}

struct Recording {
    name: String,
    corpus: String,
    audio: PathBuf,
    aus: Vec<Au>,
    /// Labelled words, which end a stretch, and stretches left out because
    /// another speaker talks over them.
    labelled: usize,
    overlapped: usize,
    unmapped: BTreeMap<String, usize>,
}

/// The TextGrids under `dir` with an audio file of the same name anywhere
/// under it too (DoReCo ships annotations and audio as separate downloads).
fn find(dir: &Path) -> Vec<(PathBuf, PathBuf)> {
    fn walk(d: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() { walk(&p, out) } else { out.push(p) }
        }
    }
    let mut all = Vec::new();
    walk(dir, &mut all);
    all.sort();
    let stem = |p: &Path| p.file_stem().map(|s| s.to_string_lossy().into_owned());
    let audio = |p: &Path| p.extension().is_some_and(|e| ["wav", "mp3", "flac", "ogg", "m4a"].contains(&e.to_string_lossy().to_lowercase().as_str()));
    all.iter()
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("textgrid")))
        .filter_map(|tg| {
            let s = stem(tg)?;
            let a = all.iter().find(|p| audio(p) && stem(p).as_deref() == Some(s.as_str()))?;
            Some((tg.clone(), a.clone()))
        })
        .collect()
}

fn load(tg: &Path, audio: &Path, max_s: Option<f64>) -> Result<Recording, String> {
    let text = std::fs::read_to_string(tg).map_err(|e| format!("{}: {e}", tg.display()))?;
    let tiers = read_textgrid(&text).map_err(|e| format!("{}: {e}", tg.display()))?;
    let name = tg.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    // doreco_<glottocode>_<file>; otherwise the folder's name.
    let corpus = name.strip_prefix("doreco_").and_then(|n| n.split('_').next()).map(String::from)
        .or_else(|| tg.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    let tier = |kind: &str, spk: &str| tiers.iter().find(|(n, _)| *n == format!("{kind}@{spk}")).map(|(_, t)| t.as_slice());
    let speakers: Vec<&str> = tiers.iter().filter_map(|(n, _)| n.strip_prefix("ph@")).collect();
    let mut aus: Vec<(usize, Au)> = Vec::new();
    let (mut labelled, mut unmapped) = (0, BTreeMap::new());
    let inside = |iv: &Iv, a: f64, b: f64| (iv.a + iv.b) / 2.0 > a && (iv.a + iv.b) / 2.0 < b;
    for (k, spk) in speakers.iter().enumerate() {
        let (Some(refs), Some(ph)) = (tier("ref", spk), tier("ph", spk)) else { continue };
        let wd = tier("wd", spk).unwrap_or(&[]);
        for r in refs.iter().filter(|r| !r.text.is_empty() && !r.text.starts_with('<')) {
            if max_s.is_some_and(|m| r.b > m) {
                continue;
            }
            // A labelled word (<<fm>...>, <<fs>>, <<ui>>) has no phones, so
            // the unit is cut there into runs of ordinary words, each scored
            // on its own; a pause inside a run is kept.
            let mut runs: Vec<Vec<&Iv>> = vec![Vec::new()];
            for w in wd.iter().filter(|w| inside(w, r.a, r.b)) {
                if w.text.starts_with("<<") {
                    labelled += 1;
                    runs.push(Vec::new());
                } else if !w.text.starts_with('<') && !w.text.trim().is_empty() {
                    runs.last_mut().unwrap().push(w);
                }
            }
            for ws in runs.into_iter().filter(|ws| !ws.is_empty()) {
                let (a, b) = (ws[0].a, ws[ws.len() - 1].b);
                let ps: Vec<&Iv> = ph.iter().filter(|p| inside(p, a, b) && !p.text.starts_with('<') && !p.text.trim().is_empty()).collect();
                let mut phones = Vec::new();
                for p in ps {
                    let (ipa, bad) = to_ipa(p.text.trim());
                    for c in bad {
                        *unmapped.entry(c.to_string()).or_insert(0) += 1;
                    }
                    phones.push((p.a, ipa));
                }
                let words = ws.iter().map(|w| (w.a, w.text.trim().to_string())).collect();
                aus.push((k, Au { a, b, phones, words }));
            }
        }
    }
    // Two speakers at once: neither can be scored.
    let n = aus.len();
    let overlaps: Vec<bool> = (0..n).map(|i| (0..n).any(|j| aus[j].0 != aus[i].0 && aus[j].1.a < aus[i].1.b && aus[i].1.a < aus[j].1.b)).collect();
    let overlapped = overlaps.iter().filter(|o| **o).count();
    let mut aus: Vec<Au> = aus.into_iter().zip(overlaps).filter(|(_, o)| !o).map(|((_, a), _)| a).collect();
    aus.retain(|a| !a.phones.is_empty());
    aus.sort_by(|x, y| x.a.total_cmp(&y.a));
    Ok(Recording { name, corpus, audio: audio.to_path_buf(), aus, labelled, overlapped, unmapped })
}

/// 16 kHz mono 16-bit WAV: re-timing is asked once per stretch and
/// hashes the bytes it is sent, so they had better be small.
fn wav16(x: &[f32]) -> Vec<u8> {
    let data = (x.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + data as usize);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&16_000u32.to_le_bytes());
    w.extend_from_slice(&32_000u32.to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data.to_le_bytes());
    for v in x {
        w.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    w
}

// ------------------------------------------------------------ measures

/// Running totals for one model on one corpus.
#[derive(Default)]
struct Score {
    recordings: usize,
    aus: usize,
    speech_s: f64,
    run_s: f64,
    // free: onsets matched one to one
    n_ref: usize,
    n_hyp: usize,
    hits: usize,
    hits_wide: usize,
    /// Reference word onsets, and those with a predicted onset matched within TOL.
    n_word: usize,
    word_hits: usize,
    offsets: Vec<f64>,
    // error rate
    per_ref: usize,
    per_hyp: usize,
    per_err: usize,
    per_err_loose: usize,
    // forced alignment
    forced_err: Vec<f64>,
    /// The same onsets spread evenly over their stretch: what an aligner
    /// that heard nothing would score.
    even_err: Vec<f64>,
    forced_word_err: Vec<f64>,
    forced_unplaced: usize,
    forced_failed: usize,
    /// Reference characters the model has no symbol for.
    missing: BTreeMap<String, usize>,
}

/// One-to-one matching of reference and predicted times, both sorted: each
/// reference takes the nearest unused prediction within `tol`. Returns which
/// references found one.
fn matched(refs: &[f64], hyps: &[f64], tol: f64) -> Vec<bool> {
    let mut used = vec![false; hyps.len()];
    let mut from = 0;
    refs.iter()
        .map(|&r| {
            while from < hyps.len() && hyps[from] < r - tol {
                from += 1;
            }
            let best = (from..hyps.len())
                .take_while(|&j| hyps[j] <= r + tol)
                .filter(|&j| !used[j])
                .min_by(|&x, &y| (hyps[x] - r).abs().total_cmp(&(hyps[y] - r).abs()));
            if let Some(j) = best {
                used[j] = true;
            }
            best.is_some()
        })
        .collect()
}

fn edit_distance<T: PartialEq>(r: &[T], h: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rc != hc)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()]
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    Some(if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 })
}

/// The characters a model's symbols are written with.
fn vocab(tokens: &Path) -> HashSet<char> {
    std::fs::read_to_string(tokens).unwrap_or_default().lines()
        .filter_map(|l| l.rsplit_once(' ').map(|(s, _)| s))
        .filter(|s| !s.starts_with('<'))
        .flat_map(|s| s.chars().collect::<Vec<_>>())
        .flat_map(|c| c.to_lowercase().chain(std::iter::once(c)).collect::<Vec<_>>())
        .collect()
}

/// What the model read: onsets (phones for ZIPA, with a mark joined to the
/// letter before it; words for the others) and, for ZIPA, the phone units.
fn reading(v: &Value, phones: bool) -> Vec<(f64, String)> {
    let mut out: Vec<(f64, String)> = Vec::new();
    for cue in v["cues"].as_array().into_iter().flatten() {
        for w in cue["words"].as_array().into_iter().flatten() {
            if !phones {
                if let (Some(a), Some(t)) = (w["start"].as_f64(), w["text"].as_str()) {
                    out.push((a, t.to_string()));
                }
                continue;
            }
            let (Some(ps), Some(ts)) = (w["phones"].as_array(), w["phone_times"].as_array()) else { continue };
            for (p, t) in ps.iter().zip(ts) {
                let (Some(p), Some(t)) = (p.as_str(), t.as_f64()) else { continue };
                match out.last_mut() {
                    Some(last) if !p.is_empty() && p.chars().all(is_mark) => last.1.push_str(p),
                    _ => out.push((t, p.to_string())),
                }
            }
        }
    }
    out
}

fn score_recording(local: &Local, model: &str, phones: &HashSet<char>, rec: &Recording, max_s: Option<f64>, s: &mut Score) -> Result<(), String> {
    let is_phones = models::find(model).is_some_and(|e| e.kind == Kind::Phones);
    let bytes = std::fs::read(&rec.audio).map_err(|e| format!("{}: {e}", rec.audio.display()))?;
    let mut x = audio::decode_16k_mono(bytes, &rec.audio.to_string_lossy())?;
    if let Some(m) = max_s {
        x.truncate((m * audio::WHISPER_RATE as f64) as usize);
    }
    let wav = wav16(&x);
    drop(x);
    let t = Instant::now();
    let (code, v) = local.transcribe(Request {
        model: model.into(), bytes: wav.clone(), filename: "eval.wav".into(),
        language: None, speakers: None, threshold: None,
    });
    if code != 200 {
        return Err(v["error"].as_str().unwrap_or("failed").to_string());
    }
    let read = reading(&v, is_phones);
    s.recordings += 1;
    s.aus += rec.aus.len();
    s.speech_s += rec.aus.iter().map(|a| a.b - a.a).sum::<f64>();

    // Free reading: onsets inside the scored stretches (a little before each,
    // so an early onset of its first phone still counts).
    let within = |t: f64| rec.aus.iter().any(|a| t >= a.a - TOL_WIDE && t <= a.b);
    let hyps: Vec<f64> = read.iter().map(|r| r.0).filter(|&t| within(t)).collect();
    let mut refs: Vec<(f64, bool)> = Vec::new();
    for au in &rec.aus {
        let word_starts: Vec<f64> = au.words.iter().map(|w| w.0).collect();
        if is_phones {
            refs.extend(au.phones.iter().map(|p| (p.0, word_starts.iter().any(|w| (w - p.0).abs() < 1e-3))));
        } else {
            refs.extend(word_starts.iter().map(|&w| (w, true)));
        }
    }
    refs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let ref_t: Vec<f64> = refs.iter().map(|r| r.0).collect();
    let m = matched(&ref_t, &hyps, TOL);
    s.n_ref += refs.len();
    s.n_hyp += hyps.len();
    s.hits += m.iter().filter(|x| **x).count();
    s.hits_wide += matched(&ref_t, &hyps, TOL_WIDE).iter().filter(|x| **x).count();
    s.n_word += refs.iter().filter(|r| r.1).count();
    s.word_hits += refs.iter().zip(&m).filter(|(r, x)| r.1 && **x).count();
    for &r in &ref_t {
        if let Some(h) = hyps.iter().min_by(|a, b| (*a - r).abs().total_cmp(&(*b - r).abs())).filter(|h| (*h - r).abs() <= NEAR) {
            s.offsets.push(h - r);
        }
    }

    // Phone error rate, stretch by stretch.
    if is_phones {
        for au in &rec.aus {
            let r: Vec<String> = au.phones.iter().flat_map(|p| units(&p.1)).collect();
            let h: Vec<String> = read.iter().filter(|p| p.0 >= au.a - TOL_WIDE && p.0 < au.b).flat_map(|p| units(&p.1)).collect();
            let base = |v: &[String]| v.iter().map(|u| u.chars().next().unwrap_or(' ')).collect::<Vec<char>>();
            s.per_ref += r.len();
            s.per_hyp += h.len();
            s.per_err += edit_distance(&r, &h);
            s.per_err_loose += edit_distance(&base(&r), &base(&h));
        }
    }

    // Forced alignment of the reference, stretch by stretch. The first
    // onset of a stretch is where the stretch starts, which the request
    // gives away, so it is not scored.
    for au in &rec.aus {
        let labels: Vec<String> = if is_phones { au.phones.iter().map(|p| p.1.clone()).collect() } else { au.words.iter().map(|w| w.1.clone()).collect() };
        let truth: Vec<f64> = if is_phones { au.phones.iter().map(|p| p.0).collect() } else { au.words.iter().map(|w| w.0).collect() };
        if labels.len() < 2 {
            continue;
        }
        for l in &labels {
            for c in l.to_lowercase().chars().filter(|c| !phones.contains(c) && !c.is_whitespace()) {
                *s.missing.entry(c.to_string()).or_insert(0) += 1;
            }
        }
        let word_starts: Vec<f64> = au.words.iter().map(|w| w.0).collect();
        match local.align(wav.clone(), "eval.wav", model, au.a, au.b, &labels) {
            Ok(v) => {
                let ivs = v["intervals"].as_array().cloned().unwrap_or_default();
                for (i, iv) in ivs.iter().enumerate().skip(1) {
                    let Some(at) = iv["start"].as_f64() else { continue };
                    if iv["conf"].is_null() {
                        s.forced_unplaced += 1;
                    }
                    let e = at - truth[i];
                    s.forced_err.push(e);
                    s.even_err.push(au.a + (au.b - au.a) * i as f64 / labels.len() as f64 - truth[i]);
                    if is_phones && word_starts.iter().any(|w| (w - truth[i]).abs() < 1e-3) {
                        s.forced_word_err.push(e);
                    }
                }
            }
            Err(_) => s.forced_failed += 1,
        }
    }
    s.run_s += t.elapsed().as_secs_f64();
    Ok(())
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn summary(corpus: &str, model: &str, s: &Score, is_phones: bool) -> Value {
    let ratio = |a: usize, b: usize| (b > 0).then(|| r3(a as f64 / b as f64));
    let (p, r) = (s.hits as f64 / s.n_hyp.max(1) as f64, s.hits as f64 / s.n_ref.max(1) as f64);
    let f1 = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    // Räsänen et al. (2009): penalises over-segmentation, which F1 forgives.
    let os = if p > 0.0 { r / p - 1.0 } else { 0.0 };
    let r1 = ((1.0 - r).powi(2) + os.powi(2)).sqrt();
    let r2 = (-os + r - 1.0) / 2f64.sqrt();
    let rval = 1.0 - (r1.abs() + r2.abs()) / 2.0;
    let within = |v: &[f64], tol: f64| (!v.is_empty()).then(|| r3(v.iter().filter(|e| e.abs() <= tol + 1e-9).count() as f64 / v.len() as f64));
    let abs = |v: &[f64]| median(&v.iter().map(|e| e.abs()).collect::<Vec<_>>()).map(|m| r3(m * 1000.0));
    let unit = if is_phones { "phone" } else { "word" };
    json!({
        "corpus": corpus, "model": model, "unit": unit,
        "recordings": s.recordings, "stretches": s.aus, "speech_seconds": r3(s.speech_s), "seconds": r3(s.run_s),
        "free": {
            "reference_onsets": s.n_ref, "predicted_onsets": s.n_hyp,
            "precision": r3(p), "recall": r3(r), "f1": r3(f1), "r_value": r3(rval),
            "recall_50ms": ratio(s.hits_wide, s.n_ref),
            "word_onset_recall": ratio(s.word_hits, s.n_word),
            "median_offset_ms": median(&s.offsets).map(|m| r3(m * 1000.0)),
        },
        "forced": {
            "onsets": s.forced_err.len(),
            "within_20ms": within(&s.forced_err, TOL), "within_50ms": within(&s.forced_err, TOL_WIDE),
            "median_abs_ms": abs(&s.forced_err),
            "median_offset_ms": median(&s.forced_err).map(|m| r3(m * 1000.0)),
            "even_spread_within_20ms": within(&s.even_err, TOL),
            "word_onsets": is_phones.then(|| json!({ "n": s.forced_word_err.len(), "within_20ms": within(&s.forced_word_err, TOL), "median_abs_ms": abs(&s.forced_word_err) })),
            "unplaced": s.forced_unplaced, "failed_stretches": s.forced_failed,
        },
        "per": is_phones.then(|| json!({ "strict": ratio(s.per_err, s.per_ref), "letters_only": ratio(s.per_err_loose, s.per_ref), "reference_units": s.per_ref, "predicted_units": s.per_hyp })),
        "missing_in_model": s.missing,
    })
}

fn add(into: &mut Score, s: &Score) {
    into.recordings += s.recordings; into.aus += s.aus; into.speech_s += s.speech_s; into.run_s += s.run_s;
    into.n_ref += s.n_ref; into.n_hyp += s.n_hyp; into.hits += s.hits; into.hits_wide += s.hits_wide;
    into.n_word += s.n_word; into.word_hits += s.word_hits; into.offsets.extend(&s.offsets);
    into.per_ref += s.per_ref; into.per_hyp += s.per_hyp; into.per_err += s.per_err; into.per_err_loose += s.per_err_loose;
    into.forced_err.extend(&s.forced_err); into.even_err.extend(&s.even_err); into.forced_word_err.extend(&s.forced_word_err);
    into.forced_unplaced += s.forced_unplaced; into.forced_failed += s.forced_failed;
    for (k, v) in &s.missing {
        *into.missing.entry(k.clone()).or_insert(0) += v;
    }
}

// ------------------------------------------------------------ the run

pub fn run(args: &[String], plan: impl Fn(&[String]) -> Result<Vec<(String, PathBuf)>, String>) -> Result<Value, String> {
    let after = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let dir = PathBuf::from(after("--eval").ok_or("--eval needs a folder")?);
    let max_s = after("--seconds").and_then(|v| v.parse::<f64>().ok());
    let ids: Vec<String> = after("--models").map_or_else(|| DEFAULT_MODELS.iter().map(|s| s.to_string()).collect(), |m| m.split(',').map(String::from).collect());
    for id in &ids {
        let ok = id.starts_with("mms-") || models::find(id).is_some_and(|e| matches!(e.kind, Kind::Phones | Kind::Letters));
        if !ok {
            return Err(format!("{id} cannot be evaluated: only ZIPA, Omnilingual and MMS place boundaries Lana can re-time"));
        }
    }
    let pairs = find(&dir);
    if pairs.is_empty() {
        return Err(format!("no TextGrid with an audio file of the same name under {}", dir.display()));
    }
    let mut recs = Vec::new();
    for (tg, a) in &pairs {
        let r = load(tg, a, max_s)?;
        let phones: usize = r.aus.iter().map(|a| a.phones.len()).sum();
        println!("{:<48} {:>4} stretches, {phones} phones ({} labelled words, {} overlaps left out){}", r.name, r.aus.len(), r.labelled, r.overlapped,
                 if r.unmapped.is_empty() { String::new() } else { format!(", unmapped X-SAMPA: {:?}", r.unmapped) });
        recs.push(r);
    }
    let plan = plan(&ids)?;

    let mut rows = Vec::new();
    println!("\n{:<10} {:<18} {:>6} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7} {:>6}", "corpus", "model", "onsets", "P20", "R20", "F20", "Rval", "off ms", "forced", "med ms", "PER");
    for (id, mdir) in &plan {
        let local = Local::new(Models::new(mdir.clone()));
        let is_phones = models::find(id).is_some_and(|e| e.kind == Kind::Phones);
        let tokens = match id.strip_prefix("mms-") {
            Some(code) => local.models.mms_lang_dir(code).join("tokens.txt"),
            None => models::find(id).map(|e| local.models.named(&e, "tokens.txt")).unwrap_or_default(),
        };
        let chars = vocab(&tokens);
        let mut by_corpus: BTreeMap<String, Score> = BTreeMap::new();
        for rec in &recs {
            let s = by_corpus.entry(rec.corpus.clone()).or_default();
            if let Err(e) = score_recording(&local, id, &chars, rec, max_s, s) {
                println!("{id} on {}: {e}", rec.name);
            }
        }
        let mut all = Score::default();
        let mut model_rows: Vec<Value> = by_corpus.iter().map(|(corpus, s)| {
            add(&mut all, s);
            summary(corpus, id, s, is_phones)
        }).collect();
        if by_corpus.len() > 1 {
            model_rows.push(summary("all", id, &all, is_phones));
        }
        for row in model_rows {
            let f = |v: &Value| v.as_f64().map_or("-".into(), |x| format!("{x:.2}"));
            let ms = |v: &Value| v.as_f64().map_or("-".into(), |x| format!("{x:.0}"));
            println!("{:<10} {:<18} {:>6} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>7} {:>6}", row["corpus"].as_str().unwrap_or(""), id,
                     row["free"]["reference_onsets"], f(&row["free"]["precision"]), f(&row["free"]["recall"]), f(&row["free"]["f1"]),
                     f(&row["free"]["r_value"]), ms(&row["free"]["median_offset_ms"]), f(&row["forced"]["within_20ms"]),
                     ms(&row["forced"]["median_abs_ms"]), f(&row["per"]["strict"]));
            rows.push(row);
        }
    }
    Ok(json!({
        "folder": dir.to_string_lossy(), "seconds_per_recording": max_s,
        "recordings": recs.iter().map(|r| json!({ "name": r.name, "corpus": r.corpus, "stretches": r.aus.len(), "labelled_words": r.labelled, "overlaps": r.overlapped, "unmapped_xsampa": r.unmapped })).collect::<Vec<_>>(),
        "tolerance_ms": TOL * 1000.0,
        "results": rows,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xsampa_reads_longest_first() {
        assert_eq!(to_ipa("k_w_>").0, "kʷʼ");
        assert_eq!(to_ipa("|\\|\\_h").0, "ǁʰ");
        assert_eq!(to_ipa("a_?\\").0, "aˤ");
        assert_eq!(to_ipa("r\\").0, "ɹ");
        assert_eq!(to_ipa("t:S").0, "tːʃ");
        assert_eq!(to_ipa("a%").1, vec!['%']);
    }

    #[test]
    fn units_keep_marks_with_their_letter() {
        assert_eq!(units("tsʼaː"), vec!["t", "sʼ", "aː"]);
    }

    #[test]
    fn matching_is_one_to_one() {
        // Two references 10 ms apart cannot both take the one prediction.
        assert_eq!(matched(&[1.0, 1.01], &[1.005], 0.02), vec![true, false]);
        assert_eq!(matched(&[1.0, 2.0], &[1.019, 2.03], 0.02), vec![true, false]);
    }

    #[test]
    fn textgrid_long_format() {
        let tg = "File type = \"ooTextFile\"\nObject class = \"TextGrid\"\n\nxmin = 0\nxmax = 1\ntiers? <exists>\nsize = 1\nitem []:\n\titem [1]:\n\t\tclass = \"IntervalTier\"\n\t\tname = \"ph@A\"\n\t\txmin = 0\n\t\txmax = 1\n\t\tintervals: size = 2\n\t\tintervals [1]:\n\t\t\txmin = 0\n\t\t\txmax = 0.5\n\t\t\ttext = \"<p:>\"\n\t\tintervals [2]:\n\t\t\txmin = 0.5\n\t\t\txmax = 1\n\t\t\ttext = \"say \"\"a\"\"\"\n";
        let t = read_textgrid(tg).unwrap();
        assert_eq!(t[0].0, "ph@A");
        assert_eq!(t[0].1.len(), 2);
        assert_eq!(t[0].1[1].text, "say \"a\"");
        assert_eq!(t[0].1[0].a, 0.0);
    }
}
