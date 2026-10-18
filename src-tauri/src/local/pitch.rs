// SPDX-License-Identifier: GPL-3.0-or-later
// A second f0 track beside Praat's, from CREPE (Kim et al. 2018, MIT), a
// network that reads pitch from the waveform rather than from its
// autocorrelation. Where the two agree the f0 is safe; where they part, in
// creak, breathy voice, or noise, the recording deserves a look, which is
// the use Lana puts it to. The ONNX exports are onnxcrepe's (MIT).
//
// Each 10 ms the network sees 64 ms of audio and scores 360 pitches, 20
// cents apart from 32 Hz to 1975 Hz. The pitch is the score-weighted mean of
// the nine around the path a Viterbi pass picks through the scores, which
// keeps it from jumping an octave between frames, and the frame's highest
// score is how periodic it sounded: CREPE's voicing confidence.

use std::{path::Path, sync::Mutex};

use ort::{session::Session, value::Tensor};

pub const HOP_S: f64 = 0.01;
const WINDOW: usize = 1024;
const HOP: usize = 160;
const BINS: usize = 360;
/// Frames per run of the network.
const BATCH: usize = 512;
/// The furthest the path may move between frames, in bins (12 = 240 cents
/// in 10 ms), as in CREPE's own Viterbi decoding.
const STEP: usize = 12;

pub struct Crepe {
    session: Mutex<Session>,
}

pub struct Track {
    pub f0: Vec<f32>,
    pub conf: Vec<f32>,
}

fn cents(bin: f64) -> f64 {
    1997.379_408_437_619_1 + 20.0 * bin
}

impl Crepe {
    pub fn load(model: &Path, threads: i32) -> Result<Self, String> {
        let session = Session::builder(super::frames::environment()?)
            .and_then(|b| b.with_intra_threads(threads.max(1) as usize).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(model))
            .map_err(|e| format!("could not load {}: {e}", model.display()))?;
        Ok(Crepe { session: Mutex::new(session) })
    }

    /// f0 in Hz and periodicity every 10 ms of 16 kHz mono audio, frame i
    /// centred on i × 10 ms. `on_progress` gets the share done and returns
    /// false to stop.
    pub fn track(&self, x: &[f32], mut on_progress: impl FnMut(f64) -> bool) -> Result<Track, String> {
        let n = x.len() / HOP + 1;
        let mut scores = Vec::with_capacity(n * BINS);
        let err = |e: ort::Error| format!("the pitch model failed: {e}");
        let mut session = self.session.lock().unwrap();
        for first in (0..n).step_by(BATCH) {
            let count = BATCH.min(n - first);
            let mut frames = Vec::with_capacity(count * WINDOW);
            for f in first..first + count {
                // Centred frames, zero beyond the ends, each normalised to
                // zero mean and unit spread as CREPE was trained.
                let start = (f * HOP) as isize - (WINDOW / 2) as isize;
                let w: Vec<f32> = (0..WINDOW as isize)
                    .map(|i| x.get((start + i) as usize).copied().filter(|_| start + i >= 0).unwrap_or(0.0))
                    .collect();
                let mean = w.iter().sum::<f32>() / WINDOW as f32;
                let sd = (w.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / WINDOW as f32).sqrt().max(1e-10);
                frames.extend(w.iter().map(|v| (v - mean) / sd));
            }
            let t = Tensor::from_array(([count, WINDOW], frames)).map_err(err)?;
            let out = session.run(ort::inputs!["frames" => t]).map_err(err)?;
            let (_, p) = out[0].try_extract_tensor::<f32>().map_err(err)?;
            scores.extend_from_slice(&p[..count * BINS]);
            if !on_progress((first + count) as f64 / n as f64) {
                return Err("stopped".into());
            }
        }
        let path = viterbi(&scores, n);
        let mut f0 = Vec::with_capacity(n);
        let mut conf = Vec::with_capacity(n);
        for (t, &c) in path.iter().enumerate() {
            let row = &scores[t * BINS..(t + 1) * BINS];
            let (lo, hi) = (c.saturating_sub(4), (c + 5).min(BINS));
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for (b, &s) in row.iter().enumerate().take(hi).skip(lo) {
                num += s as f64 * cents(b as f64);
                den += s as f64;
            }
            let c = if den > 0.0 { num / den } else { cents(c as f64) };
            f0.push((10.0 * 2f64.powf(c / 1200.0)) as f32);
            conf.push(row.iter().cloned().fold(0.0, f32::max));
        }
        Ok(Track { f0, conf })
    }
}

/// The likeliest path of pitch bins through the frames' scores, moving at
/// most STEP bins a frame, the nearer the likelier.
fn viterbi(scores: &[f32], n: usize) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let trans: Vec<f32> = (0..=STEP).map(|d| ((STEP + 1 - d) as f32).ln()).collect();
    let obs = |t: usize, b: usize| {
        let row = &scores[t * BINS..(t + 1) * BINS];
        let sum: f32 = row.iter().sum::<f32>().max(1e-12);
        (row[b] / sum).max(1e-12).ln()
    };
    let mut d: Vec<f32> = (0..BINS).map(|b| obs(0, b)).collect();
    let mut back = vec![0u16; n * BINS];
    for t in 1..n {
        let mut next = vec![f32::NEG_INFINITY; BINS];
        for (b, slot) in next.iter_mut().enumerate() {
            let (lo, hi) = (b.saturating_sub(STEP), (b + STEP).min(BINS - 1));
            let mut best = (f32::NEG_INFINITY, b);
            for (p, &dp) in d.iter().enumerate().take(hi + 1).skip(lo) {
                let v = dp + trans[p.abs_diff(b)];
                if v > best.0 {
                    best = (v, p);
                }
            }
            *slot = best.0 + obs(t, b);
            back[t * BINS + b] = best.1 as u16;
        }
        d = next;
    }
    let mut b = (0..BINS).max_by(|&i, &j| d[i].total_cmp(&d[j])).unwrap();
    let mut path = vec![0; n];
    for t in (0..n).rev() {
        path[t] = b;
        if t > 0 {
            b = back[t * BINS + b] as usize;
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_path_follows_a_glide_and_ignores_an_octave_blip() {
        // A peak gliding up one bin a frame, with one frame that scores an
        // octave (60 bins) up a little higher.
        let n = 20;
        let mut s = vec![0.01f32; n * BINS];
        for t in 0..n {
            s[t * BINS + 100 + t] = 0.9;
        }
        s[10 * BINS + 170] = 0.95;
        let p = viterbi(&s, n);
        assert_eq!(p[10], 110);
        assert_eq!(p[19], 119);
    }

    /// The demo clips' f0 against Praat's (median 117 Hz German, 205 Hz
    /// Georgian, from the voice-quality work):
    /// cargo test --release -- --ignored crepe_on_demo --nocapture
    #[test]
    #[ignore]
    fn crepe_on_demo() {
        for size in ["tiny", "small", "medium", "full"] {
            let c = Crepe::load(Path::new(&format!("/mnt/d/Models/lana-eval/crepe/{size}.onnx")), 6).unwrap();
            for clip in ["de-vinson", "ka-vinson"] {
                let x = crate::local::audio::decode_16k_mono(std::fs::read(format!("../web/demo/{clip}.wav")).unwrap(), "a.wav").unwrap();
                let t0 = std::time::Instant::now();
                let tr = c.track(&x, |_| true).unwrap();
                let took = t0.elapsed().as_secs_f64();
                let mut v: Vec<f32> = tr.f0.iter().zip(&tr.conf).filter(|(_, &c)| c > 0.5).map(|(f, _)| *f).collect();
                v.sort_by(f32::total_cmp);
                println!("{size} {clip}: {:.1}x real time, {} of {} frames voiced, median {:.0} Hz, p10 {:.0}, p90 {:.0}",
                    x.len() as f64 / 16_000.0 / took, v.len(), tr.f0.len(), v[v.len() / 2], v[v.len() / 10], v[v.len() * 9 / 10]);
            }
        }
    }
}
