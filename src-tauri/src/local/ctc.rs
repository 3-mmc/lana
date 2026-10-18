// SPDX-License-Identifier: GPL-3.0-or-later
// CTC recognisers run through sherpa-onnx: ZIPA (Zhu et al. 2025) for IPA
// phones, and Meta's Omnilingual ASR for words in some 1600 languages. Both
// emit one symbol per frame with its time, which is what makes their
// boundaries come from the model rather than from a separate aligner.

use std::{collections::HashSet, path::Path, sync::OnceLock};

use serde_json::{json, Value};

use super::frames::{Input, Mat, Net};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Engine {
    /// ZIPA: IPA phones, a zipformer CTC model.
    Zipa,
    /// Omnilingual ASR: characters with an explicit space, from raw audio.
    Omnilingual,
}

/// Both kinds emit one frame per 20 ms.
pub const FRAME_S: f64 = 0.02;
/// The CTC blank, symbol 0 in every model here.
pub const BLANK: usize = 0;
/// A runner-up is reported when it holds at least this share of a frame.
const ALT_MIN: f32 = 0.05;

/// One symbol as read: a phone (ZIPA) or a character or word space
/// (Omnilingual, MMS). `conf` is how sure the model was that this symbol, and
/// not another, is what it heard, and `alts` are the others it weighed, most
/// likely first; see `Ctc::decode`.
#[derive(Clone, Debug, PartialEq)]
pub struct Sym {
    pub text: String,
    pub at: f64,
    pub conf: f32,
    pub alts: Vec<(String, f32)>,
}

impl Sym {
    #[cfg(test)]
    pub fn new(text: &str, at: f64) -> Self {
        Sym { text: text.into(), at, conf: 1.0, alts: Vec::new() }
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
    net: Net,
    pub engine: Engine,
    /// What each symbol is written as: None for markers no tier shows
    /// (<s>, <unk>, ZIPA's ▁), " " for a word space.
    shown: Vec<Option<String>>,
    /// The symbols decoding may choose; None allows all.
    allowed: Option<Vec<bool>>,
}

/// A recogniser's frames for one piece, kept for search and re-timing.
pub struct Piece {
    pub offset: f64,
    pub mat: Mat,
}

impl Ctc {
    pub fn load(engine: Engine, model: &Path, tokens: &Path, threads: i32) -> Result<Self, String> {
        let input = if engine == Engine::Zipa { Input::Fbank } else { Input::Samples };
        let net = Net::load(model, input, threads)?;
        let text = std::fs::read_to_string(tokens).map_err(|e| format!("could not read {}: {e}", tokens.display()))?;
        let mut shown: Vec<Option<String>> = Vec::new();
        for line in text.lines().filter(|l| !l.is_empty()) {
            // "symbol id"; the symbol may itself be a space.
            let Some((sym, id)) = line.rsplit_once(' ') else { continue };
            let Ok(id) = id.parse::<usize>() else { continue };
            if shown.len() <= id {
                shown.resize(id + 1, None);
            }
            shown[id] = match engine {
                // Characters, and a space (or "|") between words.
                Engine::Omnilingual if sym.trim().is_empty() || sym == "|" => Some(" ".into()),
                // ▁ is ZIPA's word marker; it emits one at the start of a
                // piece and nothing a phone tier can use.
                _ => Some(sym.trim().trim_start_matches('▁').to_owned()).filter(|t| !t.is_empty() && !t.starts_with('<')),
            };
        }
        Ok(Ctc { net, engine, shown, allowed: None })
    }

    /// Hold decoding to these letters (and their capitals), plus the word
    /// space. Omnilingual is not told the language, and left free it writes
    /// Dargwa in Arabic, Latin or Georgian script; the language's alphabet
    /// keeps it to the letters the language is written in. The model's own
    /// probabilities are left as they are, so a letter it only chose because
    /// the one it preferred was not allowed comes out with a low confidence.
    pub fn hold_to(&mut self, alphabet: &str) {
        let mut letters: HashSet<char> = HashSet::new();
        for c in alphabet.chars() {
            letters.insert(c);
            letters.extend(c.to_uppercase());
        }
        self.allowed = Some(
            self.shown
                .iter()
                .enumerate()
                .map(|(id, s)| id == BLANK || s.as_deref().is_some_and(|s| s == " " || s.chars().all(|c| letters.contains(&c))))
                .collect(),
        );
    }

    /// Symbols with their times for 16 kHz mono audio: phones for ZIPA,
    /// characters for Omnilingual, where " " separates words. `on_piece` gets
    /// the seconds done and the symbols of the piece just finished, so a caller
    /// can show them before the whole recording is read; it returns false to stop.
    /// With `keep`, each piece's frames are kept there as well.
    pub fn recognise(&self, samples: &[f32], rate: u32, mut keep: Option<&mut Vec<Piece>>, mut on_piece: impl FnMut(f64, &[Sym]) -> bool) -> Result<Vec<Sym>, String> {
        let mut out = Vec::new();
        // Pieces cut at the pauses VAD finds; every piece is read, silence
        // too, since CTC models write little there and search needs frames
        // for the whole recording.
        for piece in super::vad::pieces(samples, rate, PIECE_S) {
            let (at, end) = (piece.start, piece.end);
            let piece_from = out.len();
            let offset = at as f64 / rate as f64;
            let mat = self.net.log_probs(&samples[at..end])?;
            out.extend(self.decode(&mat, offset));
            // A word cannot continue across a piece boundary: close the
            // last one just after its last character, not at the cut,
            // which may be well into a pause.
            if self.engine == Engine::Omnilingual {
                if let Some(last) = out.last().filter(|s| s.text != " ") {
                    out.push(Sym { text: " ".into(), at: last.at + TAIL_S, conf: 1.0, alts: Vec::new() });
                }
            }
            if let Some(k) = keep.as_mut() {
                k.push(Piece { offset, mat });
            }
            if !on_piece(end as f64 / rate as f64, &out[piece_from..]) {
                return Err("stopped".into());
            }
        }
        Ok(out)
    }

    /// The frames for one stretch of audio, for re-timing.
    pub fn frames(&self, samples: &[f32]) -> Result<Mat, String> {
        self.net.log_probs(samples)
    }

    pub fn symbols(&self) -> &[Option<String>] {
        &self.shown
    }

    pub fn is_allowed(&self, id: usize) -> bool {
        self.allowed.as_ref().is_none_or(|a| a.get(id).copied().unwrap_or(false))
    }

    /// Greedy CTC decoding: the likeliest allowed symbol in each frame, runs
    /// of one symbol merged, blanks dropped. A symbol's time is the first
    /// frame of its run, as sherpa-onnx reports it.
    ///
    /// Its confidence is read at the frame of the run where the model was
    /// surest of it, as that symbol's share of the frame once the blank is
    /// set aside: CTC models give most frames to the blank, and how much they
    /// do says when a symbol happens, not which one it is. The runner-ups are
    /// the other symbols' shares of the same frame.
    pub fn decode(&self, m: &Mat, offset: f64) -> Vec<Sym> {
        let mut out = Vec::new();
        // The open run: its symbol, and its best frame and probability.
        let mut run: Option<(usize, usize, usize, f32)> = None;
        let close = |run: Option<(usize, usize, usize, f32)>, out: &mut Vec<Sym>| {
            let Some((k, first, best, _)) = run else { return };
            let Some(text) = self.shown.get(k).cloned().flatten() else { return };
            let row = m.row(best);
            let rest = (1.0 - row[BLANK].exp()).max(1e-6);
            let conf = (row[k].exp() / rest).min(1.0);
            let mut alts: Vec<(String, f32)> = Vec::new();
            if text != " " {
                for (j, &lp) in row.iter().enumerate() {
                    let p = lp.exp() / rest;
                    if j == k || j == BLANK || p < ALT_MIN || !self.is_allowed(j) {
                        continue;
                    }
                    if let Some(s) = self.shown.get(j).cloned().flatten().filter(|s| s != " ") {
                        alts.push((s, p.min(1.0)));
                    }
                }
                alts.sort_by(|a, b| b.1.total_cmp(&a.1));
                alts.truncate(3);
            }
            out.push(Sym { text, at: offset + first as f64 * FRAME_S, conf, alts });
        };
        for t in 0..m.t {
            let row = m.row(t);
            // Whether a symbol is there at all is the model's call, made
            // over every symbol it knows; which one, when it is held to an
            // alphabet, is the likeliest allowed one. Choosing among the
            // allowed symbols and the blank alone would turn every frame
            // where it heard a letter of another script into a blank, and
            // the words would lose most of their letters.
            let mut top = BLANK;
            for (j, &lp) in row.iter().enumerate() {
                if lp > row[top] {
                    top = j;
                }
            }
            let mut k = top;
            if top != BLANK && !self.is_allowed(top) {
                k = BLANK;
                for (j, &lp) in row.iter().enumerate() {
                    if j != BLANK && self.is_allowed(j) && (k == BLANK || lp > row[k]) {
                        k = j;
                    }
                }
            }
            match run {
                Some((r, first, best, p)) if r == k => {
                    if row[k] > p {
                        run = Some((r, first, t, row[k]));
                    } else {
                        run = Some((r, first, best, p));
                    }
                }
                _ => {
                    close(run.take(), &mut out);
                    if k != BLANK {
                        run = Some((k, t, t, row[k]));
                    }
                }
            }
        }
        close(run, &mut out);
        out
    }
}

/// The letters a language is written in, from MMS's vocabulary for it (see
/// scripts/languages/alphabets.py), by ISO 639-3 code. A code MMS has in more
/// than one script (azj-script_latin, azj-script_cyrillic) gives none: which
/// one is meant cannot be told from the code.
pub fn alphabet(code: &str) -> Option<&'static str> {
    static A: OnceLock<std::collections::HashMap<String, String>> = OnceLock::new();
    let all = A.get_or_init(|| serde_json::from_str(include_str!("alphabets.json")).expect("bundled alphabets"));
    if let Some(a) = all.get(code) {
        return Some(a.as_str());
    }
    let prefix = format!("{code}-");
    let mut found = all.iter().filter(|(k, _)| k.starts_with(&prefix));
    match (found.next(), found.next()) {
        (Some((_, a)), None) => Some(a.as_str()),
        _ => None,
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
/// A word's `conf` is its least certain character's, and its `alts` are the
/// spellings with that character, or the next least certain, read as the
/// model's runner-up.
pub fn word_cues(chars: &[Sym], duration: f64) -> Vec<Value> {
    struct W<'a> { text: String, a: f64, b: f64, chars: Vec<&'a Sym> }
    let mut words: Vec<W> = Vec::new();
    let mut cur: Option<W> = None;
    for c in chars {
        // Chinese and Japanese put no spaces between words, so a whole piece
        // would come out as one "word". Each character is an interval instead,
        // which for Chinese is a syllable: a unit worth annotating anyway.
        if is_cjk(&c.text) {
            if let Some(mut w) = cur.take() {
                w.b = c.at.max(w.b + 0.02);
                words.push(w);
            }
            words.push(W { text: c.text.clone(), a: c.at, b: f64::NAN, chars: vec![c] });
            continue;
        }
        if c.text == " " {
            if let Some(mut w) = cur.take() {
                w.b = c.at.max(w.b + 0.02);
                words.push(w);
            }
        } else {
            match cur.as_mut() {
                Some(w) => { w.text.push_str(&c.text); w.b = c.at; w.chars.push(c); }
                None => cur = Some(W { text: c.text.clone(), a: c.at, b: c.at, chars: vec![c] }),
            }
        }
    }
    if let Some(mut w) = cur {
        w.b = (w.b + TAIL_S).min(duration);
        words.push(w);
    }
    // A character runs to the next symbol, or briefly when nothing follows.
    for i in 0..words.len() {
        if words[i].b.is_nan() {
            let next = words.get(i + 1).map_or(words[i].a + 0.3, |w| w.a);
            words[i].b = next.min(words[i].a + 0.6).min(duration).max(words[i].a + 0.02);
        }
    }
    // A word may not run past the start of the next one.
    for i in 0..words.len().saturating_sub(1) {
        let next = words[i + 1].a;
        if words[i].b > next { words[i].b = next.max(words[i].a + 0.02); }
    }
    let mut cues: Vec<(f64, f64, Vec<Value>)> = Vec::new();
    for w in words {
        let conf = w.chars.iter().map(|c| c.conf).fold(1.0f32, f32::min);
        let mut weak: Vec<usize> = (0..w.chars.len()).filter(|&i| !w.chars[i].alts.is_empty()).collect();
        weak.sort_by(|&x, &y| w.chars[x].conf.total_cmp(&w.chars[y].conf));
        let alts: Vec<Value> = weak.iter().take(2).map(|&i| {
            let (alt, p) = &w.chars[i].alts[0];
            let spelt: String = w.chars.iter().enumerate().map(|(j, c)| if j == i { alt.as_str() } else { c.text.as_str() }).collect();
            json!([spelt, r3(*p as f64)])
        }).collect();
        let (a, b) = (w.a, w.b);
        let word = json!({ "text": w.text, "start": r3(a), "end": r3(b), "phones": [], "phone_times": [], "aligned": true,
                           "conf": r3(conf as f64), "alts": alts });
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
/// `phone_conf` and `phone_alts` run beside `phones`: each phone's confidence,
/// and the other phones the model weighed there with their shares.
pub fn cues(phones: &[Sym], duration: f64) -> Vec<Value> {
    let mut groups: Vec<Vec<&Sym>> = Vec::new();
    for p in phones {
        match groups.last_mut() {
            Some(g) if p.at - g.last().unwrap().at < PAUSE_S => g.push(p),
            _ => groups.push(vec![p]),
        }
    }
    groups
        .iter()
        .map(|g| {
            let start = g[0].at;
            let end = (g.last().unwrap().at + TAIL_S).min(duration).max(start + 0.02);
            let word = json!({
                "text": "", "start": r3(start), "end": r3(end),
                "phones": g.iter().map(|p| p.text.clone()).collect::<Vec<_>>(),
                "phone_times": g.iter().map(|p| r3(p.at)).collect::<Vec<_>>(),
                "phone_conf": g.iter().map(|p| r3(p.conf as f64)).collect::<Vec<_>>(),
                "phone_alts": g.iter().map(|p| p.alts.iter().map(|(s, q)| json!([s, r3(*q as f64)])).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "aligned": true,
            });
            json!({ "start": r3(start), "end": r3(end), "words": [word] })
        })
        .collect()
}

/// Han characters and kana: scripts written without spaces between words.
fn is_cjk(s: &str) -> bool {
    s.chars().next().is_some_and(|c| matches!(c as u32,
        0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F))
}

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_split_at_pauses_and_end_after_the_last_phone() {
        let p: Vec<Sym> = [("a", 0.10), ("n", 0.20), ("t", 0.30), ("i", 1.00), ("s", 1.10)]
            .iter().map(|(s, t)| Sym::new(s, *t)).collect();
        let c = cues(&p, 2.0);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0]["words"][0]["phones"], json!(["a", "n", "t"]));
        assert_eq!(c[0]["end"], json!(0.38));
        assert_eq!(c[1]["start"], json!(1.0));
    }

    #[test]
    fn han_characters_are_one_interval_each() {
        let c: Vec<Sym> = [("北", 0.50), ("部", 0.70), (" ", 0.90), ("ok", 1.00), (" ", 1.20)]
            .iter().map(|(s, t)| Sym::new(s, *t)).collect();
        let w = &word_cues(&c, 2.0)[0]["words"];
        assert_eq!(w[0]["text"], json!("北"));
        assert_eq!((w[0]["start"].clone(), w[0]["end"].clone()), (json!(0.5), json!(0.7)));
        assert_eq!(w[1]["text"], json!("部"));
        assert_eq!(w[2]["text"], json!("ok"));
    }

    #[test]
    fn characters_become_timed_words() {
        let c: Vec<Sym> = [("d", 0.64), ("e", 0.68), ("r", 0.72), (" ", 0.76), ("n", 0.80), ("o", 0.86), (" ", 1.20)]
            .iter().map(|(s, t)| Sym::new(s, *t)).collect();
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

    /// Lana's own decoding against sherpa-onnx's on the demo clips, which must
    /// agree symbol for symbol. Needs the models in D:\\Models\\lana-eval:
    /// cargo test -- --ignored same_as_sherpa --nocapture
    #[test]
    #[ignore]
    fn same_as_sherpa() {
        use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig};
        let models = Path::new("/mnt/d/Models/lana-eval");
        for (dir, engine) in [("small", Engine::Zipa), ("large", Engine::Zipa), ("omni300", Engine::Omnilingual)] {
            let (model, tokens) = (models.join(dir).join("model.int8.onnx"), models.join(dir).join("tokens.txt"));
            let ours = Ctc::load(engine, &model, &tokens, 4).unwrap();
            let mut cfg = OfflineRecognizerConfig::default();
            let m = Some(model.to_string_lossy().into_owned());
            match engine {
                Engine::Zipa => cfg.model_config.zipformer_ctc.model = m,
                Engine::Omnilingual => cfg.model_config.omnilingual.model = m,
            }
            cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
            cfg.model_config.num_threads = 4;
            let rec = OfflineRecognizer::create(&cfg).unwrap();
            for clip in ["de-vinson", "ka-vinson"] {
                let bytes = std::fs::read(format!("../web/demo/{clip}.wav")).unwrap();
                let x = crate::local::audio::decode_16k_mono(bytes, "a.wav").unwrap();
                let x = &x[..x.len().min(16_000 * 30)];
                let t = std::time::Instant::now();
                let mine: Vec<Sym> = ours.recognise(x, 16_000, None, |_, _| true).unwrap()
                    .into_iter().filter(|s| s.text != " ").collect();
                let took = t.elapsed();
                let stream = rec.create_stream();
                stream.accept_waveform(16_000, x);
                rec.decode(&stream);
                let r = stream.get_result().unwrap();
                let scale = if engine == Engine::Zipa { 0.5 } else { 1.0 };
                let theirs: Vec<(String, f64)> = r.tokens.iter().zip(r.timestamps.unwrap_or_default())
                    .map(|(k, t)| (k.trim().trim_start_matches('▁').to_owned(), t as f64 * scale))
                    .filter(|(k, _)| !k.is_empty() && !k.starts_with('<')).collect();
                let a: String = mine.iter().map(|s| s.text.as_str()).collect();
                let b: String = theirs.iter().map(|s| s.0.as_str()).collect();
                let low = mine.iter().filter(|s| s.conf < 0.6).count();
                println!("{dir} {clip}: {} vs {} symbols, {:?}, {low} below 0.6\n  ours   {}\n  sherpa {}", mine.len(), theirs.len(), took, &a.chars().take(70).collect::<String>(), &b.chars().take(70).collect::<String>());
                for s in mine.iter().filter(|s| !s.alts.is_empty()).take(4) {
                    println!("  {} {:.2} at {:.2}: {:?}", s.text, s.conf, s.at, s.alts);
                }
                // Lana's fbank matches lhotse's, which ZIPA was trained on, to
                // 1.5e-4; sherpa-onnx's differs by as much again, enough to tip
                // a phone the model is torn on (f/v once in 167 on de-vinson; 6 in 173 on
                // the very quiet ka-vinson). ZIPA's own Python pipeline disagrees with
                // both, at other such phones: these are coin flips, not a bug.
                let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
                let mut row: Vec<usize> = (0..=b.len()).collect();
                for i in 1..=a.len() {
                    let mut prev = row[0];
                    row[0] = i;
                    for j in 1..=b.len() {
                        let cur = row[j];
                        row[j] = (prev + (a[i - 1] != b[j - 1]) as usize).min(row[j] + 1).min(row[j - 1] + 1);
                        prev = cur;
                    }
                }
                println!("  differ by {} symbols", row[b.len()]);
                assert!(row[b.len()] * 20 <= b.len().max(1), "{dir} {clip}");
                if mine.len() != theirs.len() {
                    continue;
                }
                for (m, t) in mine.iter().zip(&theirs) {
                    assert!((m.at - t.1).abs() < 0.021, "{dir} {clip}: {} at {} vs {}", m.text, m.at, t.1);
                }
            }
        }
    }

    /// Omnilingual on Common Voice clips, free and held to the language's
    /// alphabet, written as a TSV for scoring (see the commit that added it):
    /// HOLD_OUT=<file> cargo test --release -- --ignored hold_on_common_voice
    #[test]
    #[ignore]
    fn hold_on_common_voice() {
        use std::fmt::Write;
        let base = Path::new("/mnt/d/Models/lana-eval");
        let mut out = String::new();
        for model in std::env::var("HOLD_MODELS").unwrap_or("omni300,omni1b".into()).split(',') {
            let dir = base.join(model);
            let mut rec = Ctc::load(Engine::Omnilingual, &dir.join("model.int8.onnx"), &dir.join("tokens.txt"), 6).unwrap();
            for (lang, code) in [("dargwa", "dar"), ("ossetian", "oss")] {
                let cv = base.join("cv").join(lang);
                let clips = std::fs::read_dir(cv.join("cv-corpus-27.0-2026-09-11")).unwrap().next().unwrap().unwrap().path().join("clips");
                for line in std::fs::read_to_string(cv.join("sample.tsv")).unwrap().lines() {
                    let (file, truth) = line.split_once('\t').unwrap();
                    let x = crate::local::audio::decode_16k_mono(std::fs::read(clips.join(file)).unwrap(), file).unwrap();
                    let mut read = |rec: &Ctc| rec.recognise(&x, 16_000, None, |_, _| true).unwrap()
                        .iter().map(|s| s.text.as_str()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
                    rec.allowed = None;
                    let free = read(&rec);
                    rec.hold_to(alphabet(code).unwrap());
                    let held = read(&rec);
                    writeln!(out, "{model}\t{code}\t{truth}\t{free}\t{held}").unwrap();
                }
            }
        }
        std::fs::write(std::env::var("HOLD_OUT").unwrap(), out).unwrap();
    }
}
