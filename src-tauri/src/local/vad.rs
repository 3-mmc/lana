// SPDX-License-Identifier: GPL-3.0-or-later
// Where to cut a recording into the pieces the recognisers read: at pauses
// Silero VAD (MIT, 2 MB) found, rather than at the quietest 20 ms near the
// 30 s mark, and with long silences set apart so Whisper need not read them.
//
// The quietest-20-ms cut (ctc::cut_point) stays as the fallback, for when
// the VAD model is not on this computer yet or finds no pause in a stretch
// of continuous speech or music.

use std::{
    ops::Range,
    path::{Path, PathBuf},
    sync::Mutex,
};

use ort::{session::Session, value::Tensor};

use super::runtime::{self, Role};

/// Where the VAD model is, set once by Local::new. A global rather than an
/// argument so that every caller of `pieces` (Whisper here, the CTC path,
/// Parakeet) gets VAD cuts without threading a path through each of them.
static MODEL: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn use_model(path: PathBuf) {
    *MODEL.lock().unwrap() = Some(path);
}

fn model() -> Option<PathBuf> {
    MODEL.lock().unwrap().clone().filter(|p| p.is_file())
}

/// Speech is widened by this much before and after before pauses are taken
/// between it, so a cut never clips a soft onset or a trailing fricative that
/// Silero scored just under its threshold. More before than after: through
/// sherpa-onnx, Silero's onsets came late, by 0.36 s on the German demo
/// clip (it missed an unstressed "Der") and by 0.6 s on the same clip after
/// a cat; 0.15 s each side lost that word in a piece taken for silence.
const PAD_BEFORE_S: f64 = 0.6;
const PAD_AFTER_S: f64 = 0.3;
/// A pause at least this long is a piece of its own. Whisper skips pieces
/// with no speech, and silence is where it invents text ("Thank you.",
/// subtitles credits, a loop); a long pause inside a piece is read as part of
/// it. Shorter pauses are not set apart, since every Whisper call costs a
/// full 30 s window whatever its length.
const SKIP_S: f64 = 3.0;
/// A cut at a pause is not taken earlier than this share of the longest piece:
/// a 6 s piece would cost Whisper the same as a 30 s one.
const MIN_SHARE: f64 = 0.4;

/// One piece: its samples, and whether VAD heard speech in it. Without the
/// VAD model every piece counts as speech, so nothing is skipped.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub range: Range<usize>,
    pub speech: bool,
}

/// Pieces of at most `max_s` that tile the whole recording, for a caller that
/// reads every piece (the CTC models).
#[allow(dead_code)] // for the CTC path, which still cuts by ctc::cut_point
pub fn pieces(samples: &[f32], rate: u32, max_s: f64) -> Vec<Range<usize>> {
    plan(samples, rate, max_s).into_iter().map(|p| p.range).collect()
}

/// Pieces with their speech flag, for a caller that skips silent ones.
pub fn plan(samples: &[f32], rate: u32, max_s: f64) -> Vec<Piece> {
    let talk = speech(samples, rate);
    let out = split(samples, rate, max_s, talk.as_deref());
    if std::env::var_os("LANA_VAD_DEBUG").is_some() {
        let s = |r: &Range<usize>| format!("{:.2}-{:.2}", r.start as f64 / rate as f64, r.end as f64 / rate as f64);
        eprintln!("lana: VAD speech {}", talk.iter().flatten().map(s).collect::<Vec<_>>().join(" "));
        eprintln!("lana: VAD pieces {}", out.iter().map(|p| s(&p.range) + if p.speech { "" } else { " (silent)" }).collect::<Vec<_>>().join(", "));
    }
    out
}

/// The stretches Silero VAD calls speech, in samples, or None without the
/// model (or if it fails to load: the pieces then fall back to energy cuts
/// rather than failing the job). Takes about 0.1 s per minute of audio.
pub fn speech(samples: &[f32], rate: u32) -> Option<Vec<Range<usize>>> {
    let path = model()?;
    let t = std::time::Instant::now();
    let found = detect(&path, samples, rate);
    if let Some(s) = &found {
        eprintln!("lana: VAD found {} speech stretches in {:.1} s of audio in {:.2} s",
                  s.len(), samples.len() as f64 / rate as f64, t.elapsed().as_secs_f64());
    }
    found
}

fn detect(path: &Path, samples: &[f32], rate: u32) -> Option<Vec<Range<usize>>> {
    let session = match runtime::session(path, Role::Light, 1) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lana: VAD unavailable: {e}");
            return None;
        }
    };
    let mut vad = Vad::new(Silero { session, state: vec![0.0; 2 * 128], rate: rate as i64 }, rate);
    for chunk in samples.chunks(SHIFT) {
        if let Err(e) = vad.accept(chunk) {
            eprintln!("lana: VAD failed: {e}");
            return None;
        }
    }
    vad.flush();
    Some(vad.found.into_iter().map(|r| r.start..r.end.min(samples.len())).collect())
}

// Silero v5 reads 512 new samples at a time at 16 kHz (all Lana gives it,
// audio::decode_16k_mono), each with the 64 before them for context.
const SHIFT: usize = 512;
const WINDOW: usize = SHIFT + 64;
const THRESHOLD: f32 = 0.5;
/// A pause of 0.25 s is enough to cut in; shorter gaps inside a word (stop
/// closures) must not count as pauses.
const MIN_SILENCE_S: f32 = 0.25;
const MIN_SPEECH_S: f32 = 0.2;
/// A stretch of speech that runs this long is ended at the next dip that can
/// be found (a shorter pause at a higher threshold), which gives a 30 s piece
/// a place to cut in fast continuous speech too.
const MAX_SPEECH_S: f32 = 20.0;
const LONG_MIN_SILENCE_S: f32 = 0.1;
const LONG_THRESHOLD: f32 = 0.9;

/// The model and its recurrent state, carried from window to window.
struct Silero {
    session: Session,
    state: Vec<f32>,
    rate: i64,
}

impl Silero {
    fn prob(&mut self, window: &[f32]) -> Result<f32, String> {
        let err = |e: ort::Error| e.to_string();
        let x = Tensor::from_array(([1usize, window.len()], window.to_vec())).map_err(err)?;
        let state = Tensor::from_array(([2usize, 1, 128], std::mem::take(&mut self.state))).map_err(err)?;
        let sr = Tensor::from_array(([1usize], vec![self.rate])).map_err(err)?;
        let out = self.session.run(ort::inputs!["input" => x, "state" => state, "sr" => sr]).map_err(err)?;
        self.state = out["stateN"].try_extract_tensor::<f32>().map_err(err)?.1.to_vec();
        Ok(out["output"].try_extract_tensor::<f32>().map_err(err)?.1[0])
    }
}

/// sherpa-onnx's VoiceActivityDetector over Silero, which Lana used before
/// it ran models itself, and whose cuts the padding in `split` was tuned to: the same
/// hysteresis, the same widening of a stretch's start by two windows and
/// the minimum speech length, the same end at the start of the pause that
/// closed it. Sample positions count from the start of the recording; `head`
/// is where sherpa's buffer began, which a stretch never starts before.
struct Vad {
    model: Silero,
    rate: f32,
    last: Vec<f32>,
    head: usize,
    tail: usize,
    start: Option<usize>,
    triggered: bool,
    temp_start: usize,
    temp_end: usize,
    found: Vec<Range<usize>>,
}

impl Vad {
    fn new(model: Silero, rate: u32) -> Self {
        Vad { model, rate: rate as f32, last: Vec::new(), head: 0, tail: 0, start: None,
              triggered: false, temp_start: 0, temp_end: 0, found: Vec::new() }
    }

    fn samples(&self, s: f32) -> usize {
        (self.rate * s) as usize
    }

    fn accept(&mut self, chunk: &[f32]) -> Result<(), String> {
        let long = self.tail - self.head > self.samples(MAX_SPEECH_S);
        let (min_silence, threshold) = if long { (LONG_MIN_SILENCE_S, LONG_THRESHOLD) } else { (MIN_SILENCE_S, THRESHOLD) };
        let (min_silence, min_speech) = (self.samples(min_silence), self.samples(MIN_SPEECH_S));
        self.last.extend_from_slice(chunk);
        if self.last.len() < WINDOW {
            return Ok(());
        }
        let k = (self.last.len() - WINDOW) / SHIFT + 1;
        let mut speech = false;
        for i in 0..k {
            self.tail += SHIFT;
            let p = self.model.prob(&self.last[i * SHIFT..i * SHIFT + WINDOW])?;
            speech |= self.is_speech(p, threshold, min_silence, min_speech);
        }
        self.last.drain(..k * SHIFT);
        let lead = 2 * WINDOW + min_speech;
        if speech {
            self.start.get_or_insert(self.tail.saturating_sub(lead).max(self.head));
            return Ok(());
        }
        if let Some(start) = self.start.take() {
            let end = self.tail.saturating_sub(min_silence);
            if end > start {
                self.found.push(start..end);
            }
            self.head = self.head.max(end);
        } else {
            self.head = self.head.max(self.tail.saturating_sub(lead));
        }
        Ok(())
    }

    /// One window's verdict, with Silero's hysteresis: speech starts once the
    /// probability has stayed above the threshold for `min_speech` samples,
    /// and stops once it has stayed below a lower one for `min_silence`.
    fn is_speech(&mut self, p: f32, threshold: f32, min_silence: usize, min_speech: usize) -> bool {
        let at = self.tail;
        if p > threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }
        if p > threshold && self.temp_start == 0 {
            self.temp_start = at;
            return false;
        }
        if p > threshold && !self.triggered {
            if at - self.temp_start < min_speech {
                return false;
            }
            self.triggered = true;
            return true;
        }
        if p < threshold && !self.triggered {
            self.temp_start = 0;
            self.temp_end = 0;
            return false;
        }
        if p > (threshold - 0.15).max(0.01) && self.triggered {
            return true;
        }
        if p < threshold && self.triggered {
            if self.temp_end == 0 {
                self.temp_end = at;
            }
            if at - self.temp_end < min_silence {
                return true;
            }
            self.temp_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
        false
    }

    /// Speech still going at the end runs to the end.
    fn flush(&mut self) {
        if let Some(start) = self.start.take() {
            if self.tail > start {
                self.found.push(start..self.tail);
            }
        }
    }
}

/// The cutting itself, apart from the model so it can be tested. `speech` is
/// None when there is no VAD: then pieces are cut by energy alone, as before.
pub fn split(samples: &[f32], rate: u32, max_s: f64, speech: Option<&[Range<usize>]>) -> Vec<Piece> {
    let len = samples.len();
    let max = ((max_s * rate as f64) as usize).max(1);
    let Some(speech) = speech else {
        let mut out = Vec::new();
        let mut at = 0;
        while at < len {
            let end = energy_cut(samples, at, rate, max);
            out.push(Piece { range: at..end, speech: true });
            at = end;
        }
        return out;
    };
    // Speech, padded and merged; what lies between is pause.
    let (before, after) = ((PAD_BEFORE_S * rate as f64) as usize, (PAD_AFTER_S * rate as f64) as usize);
    let mut talk: Vec<Range<usize>> = Vec::new();
    for s in speech {
        let r = s.start.saturating_sub(before)..(s.end + after).min(len);
        match talk.last_mut() {
            Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
            _ => talk.push(r),
        }
    }
    // Long pauses become pieces of their own; what is left are stretches
    // that hold speech, each then cut down to `max` at the pauses inside it.
    let skip = (SKIP_S * rate as f64) as usize;
    let mut spans: Vec<(Range<usize>, bool)> = Vec::new();
    let mut at = 0;
    let mut i = 0;
    while at < len {
        // Gather speech until a long pause (or the end) follows.
        if i >= talk.len() {
            spans.push((at..len, false));
            break;
        }
        if talk[i].start >= at + skip {
            spans.push((at..talk[i].start, false));
            at = talk[i].start;
        }
        let mut end = talk[i].end;
        i += 1;
        while i < talk.len() && talk[i].start < end + skip {
            end = talk[i].end;
            i += 1;
        }
        // A short pause at the very end of the recording goes with the speech.
        let end = if len - end < skip { len } else { end };
        spans.push((at..end, true));
        at = end;
    }
    let mut out = Vec::new();
    for (span, voiced) in spans {
        if !voiced {
            // Silence costs Whisper nothing once skipped, but a CTC model
            // still reads it, so it is kept to `max` like any other piece.
            let mut a = span.start;
            while a < span.end {
                let b = (a + max).min(span.end);
                out.push(Piece { range: a..b, speech: false });
                a = b;
            }
            continue;
        }
        let mut a = span.start;
        while a < span.end {
            let b = if span.end - a <= max {
                span.end
            } else {
                pause_cut(&talk, a, a + max, (MIN_SHARE * max as f64) as usize)
                    .unwrap_or_else(|| energy_cut(samples, a, rate, max))
            };
            let voiced = talk.iter().any(|t| t.start < b && t.end > a);
            out.push(Piece { range: a..b, speech: voiced });
            a = b;
        }
    }
    out
}

/// The middle of the latest pause that ends a piece from `a` at or before
/// `limit`, at least `min` into it. A pause that runs past `limit` is cut at
/// `limit`, which is still inside it.
fn pause_cut(talk: &[Range<usize>], a: usize, limit: usize, min: usize) -> Option<usize> {
    let mut best = None;
    for w in talk.windows(2) {
        let (from, to) = (w[0].end, w[1].start);
        if to <= from || from >= limit {
            continue;
        }
        let cut = ((from + to) / 2).min(limit);
        if cut > a + min && cut > from.max(a) {
            best = Some(cut);
        }
    }
    best
}

/// The quietest-20-ms cut, held to `max` when that is under 30 s.
fn energy_cut(samples: &[f32], at: usize, rate: u32, max: usize) -> usize {
    super::ctc::cut_point(samples, at, rate).min(at + max).min(samples.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: u32 = 16_000;
    fn s(a: f64, b: f64) -> Range<usize> {
        (a * R as f64) as usize..(b * R as f64) as usize
    }
    fn secs(p: &Piece) -> (f64, f64) {
        (p.range.start as f64 / R as f64, p.range.end as f64 / R as f64)
    }

    #[test]
    fn short_audio_is_one_piece() {
        let x = vec![0.1f32; R as usize * 10];
        let p = split(&x, R, 30.0, Some(&[s(1.0, 9.0)]));
        assert_eq!(p, vec![Piece { range: 0..x.len(), speech: true }]);
    }

    #[test]
    fn long_speech_is_cut_in_the_last_pause_before_the_limit() {
        let x = vec![0.1f32; R as usize * 50];
        let talk = [s(0.0, 12.0), s(12.6, 24.0), s(25.5, 32.0), s(32.5, 50.0)];
        let p = split(&x, R, 30.0, Some(&talk));
        // The pause 24.3..24.9 (after padding) is the last before 30 s.
        let (a, b) = secs(&p[0]);
        assert_eq!(a, 0.0);
        assert!((24.5..24.7).contains(&b), "cut at {b}");
        assert!(p.iter().all(|p| p.speech));
        assert_eq!(p.last().unwrap().range.end, x.len());
    }

    #[test]
    fn long_silence_is_its_own_piece_and_has_no_speech() {
        let x = vec![0.0f32; R as usize * 40];
        let p = split(&x, R, 30.0, Some(&[s(1.0, 5.0), s(20.0, 38.0)]));
        let v: Vec<(f64, f64, bool)> = p.iter().map(|p| (secs(p).0, secs(p).1, p.speech)).collect();
        assert_eq!(v.len(), 3, "{v:?}");
        assert!(v[0].2 && !v[1].2 && v[2].2, "{v:?}");
        assert!((5.25..5.35).contains(&v[0].1));
        assert!((19.35..19.45).contains(&v[1].1));
        assert_eq!(v[2].1, 40.0);
    }

    #[test]
    fn no_speech_at_all_is_skippable() {
        let x = vec![0.0f32; R as usize * 70];
        let p = split(&x, R, 30.0, Some(&[]));
        assert_eq!(p.len(), 3);
        assert!(p.iter().all(|p| !p.speech && p.range.len() <= 30 * R as usize));
    }

    #[test]
    fn without_vad_it_falls_back_to_the_quietest_place() {
        let rate = R as usize;
        let mut x = vec![0.5f32; rate * 40];
        for v in &mut x[rate * 27..rate * 27 + 800] { *v = 0.0; }
        let p = split(&x, R, 30.0, None);
        assert_eq!(p.len(), 2);
        assert!((rate * 27..rate * 27 + 800).contains(&p[0].range.end));
        // Continuous speech with no pause VAD can use: the same fallback.
        let q = split(&x, R, 30.0, Some(&[s(0.0, 40.0)]));
        assert_eq!(q[0].range.end, p[0].range.end);
    }
}
