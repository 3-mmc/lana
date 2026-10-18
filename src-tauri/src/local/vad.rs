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

use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

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
    // Silero v5 works on 512-sample windows at 16 kHz, which is all Lana
    // gives it (audio::decode_16k_mono).
    const WINDOW: usize = 512;
    let cfg = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(path.to_string_lossy().into_owned()),
            threshold: 0.5,
            // A pause of 0.25 s is enough to cut in; shorter gaps inside a
            // word (stop closures) must not count as pauses.
            min_silence_duration: 0.25,
            min_speech_duration: 0.2,
            window_size: WINDOW as i32,
            // sherpa-onnx ends a stretch of speech that runs this long at the
            // next dip it can find, which gives a 30 s piece a place to cut
            // in fast continuous speech too.
            max_speech_duration: 20.0,
        },
        sample_rate: rate as i32,
        num_threads: 1,
        provider: Some("cpu".into()),
        ..Default::default()
    };
    // The buffer holds the stretch being read, not the recording: segments
    // are taken out as they finish, and none runs past max_speech_duration.
    let vad = VoiceActivityDetector::create(&cfg, 60.0)?;
    let mut out = Vec::new();
    let mut take = |vad: &VoiceActivityDetector| {
        while let Some(seg) = vad.front() {
            let a = seg.start().max(0) as usize;
            out.push(a..(a + seg.n().max(0) as usize).min(samples.len()));
            vad.pop();
        }
    };
    for chunk in samples.chunks(WINDOW) {
        vad.accept_waveform(chunk);
        take(&vad);
    }
    vad.flush();
    take(&vad);
    Some(out)
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
