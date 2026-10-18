// SPDX-License-Identifier: GPL-3.0-or-later
// Frame probabilities from the CTC models, which sherpa-onnx computes but
// never returns: its result is tokens and times only. Confidence, a phone's
// runner-up readings, re-timing corrected labels and search by example all
// need the probability of every symbol in every frame, so Lana runs these
// models itself, through the ONNX Runtime that sherpa-onnx already links.
//
// sherpa-onnx's static libraries carry ONNX Runtime 1.28 and export its C
// entry point, OrtGetApiBase. The `ort` crate is built without a runtime of
// its own (its `alternative-backend` feature) and is handed that one, so the
// app carries a single copy. `ort` is built for API 17, whose OrtApi struct
// is a prefix of what 1.28 provides; copying it reads nothing past the end.

use std::{path::Path, sync::{Mutex, OnceLock}};

use ort::{environment::Environment, session::Session, value::Tensor};

unsafe extern "system" {
    fn OrtGetApiBase() -> *const ort::sys::OrtApiBase;
}

/// ONNX Runtime's environment, set up on first use. sherpa-onnx creates one
/// too; the runtime keeps a single environment per process and counts users.
fn environment() -> Result<&'static Environment, String> {
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        let api = unsafe {
            let base = OrtGetApiBase();
            if base.is_null() {
                return Err("ONNX Runtime is missing from this build".into());
            }
            let api = ((*base).GetApi)(17);
            if api.is_null() {
                return Err("ONNX Runtime in this build is too old".into());
            }
            std::ptr::read(api)
        };
        ort::set_api(api);
        ort::init().with_name("lana").build().map_err(|e| format!("ONNX Runtime: {e}"))
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Log-probabilities, `t` frames of `v` symbols each, frame-major.
#[derive(Clone, Default)]
pub struct Mat {
    pub t: usize,
    pub v: usize,
    pub lp: Vec<f32>,
}

impl Mat {
    pub fn row(&self, i: usize) -> &[f32] {
        &self.lp[i * self.v..(i + 1) * self.v]
    }
}

/// What a model takes as input.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Input {
    /// 80 log mel bands every 10 ms (ZIPA); its output is already log-softmax.
    Fbank,
    /// Raw samples, normalised to zero mean and unit variance per piece
    /// (Omnilingual and MMS, as sherpa-onnx and the MMS processor do); its
    /// output is logits.
    Samples,
}

pub struct Net {
    session: Mutex<Session>,
    input: Input,
}

impl Net {
    pub fn load(model: &Path, input: Input, threads: i32) -> Result<Self, String> {
        let session = Session::builder(environment()?)
            .and_then(|b| b.with_intra_threads(threads.max(1) as usize).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(model))
            .map_err(|e| format!("could not load {}: {e}", model.display()))?;
        Ok(Net { session: Mutex::new(session), input })
    }

    /// Log-probabilities for one piece of 16 kHz mono audio.
    pub fn log_probs(&self, samples: &[f32]) -> Result<Mat, String> {
        let err = |e: ort::Error| format!("the model failed: {e}");
        let mut session = self.session.lock().unwrap();
        let mut m = match self.input {
            Input::Fbank => {
                let (frames, feats) = fbank(samples);
                if frames == 0 {
                    return Ok(Mat::default());
                }
                let x = Tensor::from_array(([1usize, frames, MEL_BINS], feats)).map_err(err)?;
                let lens = Tensor::from_array(([1usize], vec![frames as i64])).map_err(err)?;
                let out = session.run(ort::inputs!["x" => x, "x_lens" => lens]).map_err(err)?;
                let (shape, lp) = out["log_probs"].try_extract_tensor::<f32>().map_err(err)?;
                let (_, len) = out["log_probs_len"].try_extract_tensor::<i64>().map_err(err)?;
                let (t, v) = (len[0].max(0) as usize, shape[2] as usize);
                Mat { t, v, lp: lp[..t * v].to_vec() }
            }
            Input::Samples => {
                if samples.len() < 400 {
                    return Ok(Mat::default());
                }
                let x = Tensor::from_array(([1usize, samples.len()], normalised(samples))).map_err(err)?;
                let out = session.run(ort::inputs!["x" => x]).map_err(err)?;
                let (shape, logits) = out[0].try_extract_tensor::<f32>().map_err(err)?;
                let (t, v) = (shape[1] as usize, shape[2] as usize);
                Mat { t, v, lp: logits.to_vec() }
            }
        };
        if self.input == Input::Samples {
            for row in m.lp.chunks_mut(m.v.max(1)) {
                log_softmax(row);
            }
        }
        Ok(m)
    }
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
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let log = max + sum.ln();
    for v in row {
        *v -= log;
    }
}

// --------------------------------------------------------------- fbank

pub const MEL_BINS: usize = 80;
const SHIFT: usize = 160;
const LENGTH: usize = 400;
const FFT: usize = 512;

/// Kaldi's fbank as ZIPA was trained on it (lhotse's FbankConfig with 80
/// bins, no dither, snip_edges off; the same as sherpa-onnx's defaults):
/// 25 ms Povey windows every 10 ms, the DC offset removed, pre-emphasis 0.97,
/// a 512-point power spectrum and 80 triangular mel bands from 20 Hz to
/// 7600 Hz, log of each. With snip_edges off a frame is centred on its 10 ms
/// step and the signal is mirrored at both ends. Samples are in [-1, 1].
/// Returns the frame count and the features, frame-major.
pub fn fbank(x: &[f32]) -> (usize, Vec<f32>) {
    static TABLES: OnceLock<(Vec<f32>, Vec<(usize, Vec<f32>)>)> = OnceLock::new();
    let (window, banks) = TABLES.get_or_init(|| {
        let window = (0..LENGTH)
            .map(|i| {
                let a = 2.0 * std::f64::consts::PI * i as f64 / (LENGTH - 1) as f64;
                (0.5 - 0.5 * a.cos()).powf(0.85) as f32
            })
            .collect();
        (window, mel_banks())
    });
    let n = x.len();
    let frames = (n + SHIFT / 2) / SHIFT;
    let mut out = Vec::with_capacity(frames * MEL_BINS);
    let mut buf = vec![(0f32, 0f32); FFT];
    let mut w = [0f32; LENGTH];
    for f in 0..frames {
        let start = (f * SHIFT + SHIFT / 2) as isize - (LENGTH / 2) as isize;
        for (i, slot) in w.iter_mut().enumerate() {
            let mut s = start + i as isize;
            // Mirror at the edges, as Kaldi does: -1 -> 0, n -> n - 1.
            while s < 0 || s >= n as isize {
                s = if s < 0 { -s - 1 } else { 2 * n as isize - 1 - s };
            }
            *slot = x[s as usize];
        }
        let mean = w.iter().sum::<f32>() / LENGTH as f32;
        for v in w.iter_mut() {
            *v -= mean;
        }
        for i in (1..LENGTH).rev() {
            w[i] -= 0.97 * w[i - 1];
        }
        w[0] -= 0.97 * w[0];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = (if i < LENGTH { w[i] * window[i] } else { 0.0 }, 0.0);
        }
        fft(&mut buf);
        for (first, weights) in banks {
            let e: f32 = weights.iter().enumerate().map(|(k, wt)| {
                let (re, im) = buf[first + k];
                wt * (re * re + im * im)
            }).sum();
            out.push(e.max(f32::EPSILON).ln());
        }
    }
    (frames, out)
}

/// Kaldi's mel filterbank: each band as its first FFT bin and its weights.
fn mel_banks() -> Vec<(usize, Vec<f32>)> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let (lo, hi) = (mel(20.0), mel(8000.0 - 400.0));
    let delta = (hi - lo) / (MEL_BINS + 1) as f64;
    let bin_hz = 16_000.0 / FFT as f64;
    (0..MEL_BINS)
        .map(|b| {
            let (l, c, r) = (lo + b as f64 * delta, lo + (b + 1) as f64 * delta, lo + (b + 2) as f64 * delta);
            let mut first = None;
            let mut weights = Vec::new();
            for i in 0..FFT / 2 {
                let m = mel(i as f64 * bin_hz);
                if m > l && m < r {
                    first.get_or_insert(i);
                    weights.push(if m <= c { (m - l) / (c - l) } else { (r - m) / (r - c) } as f32);
                }
            }
            (first.unwrap_or(0), weights)
        })
        .collect()
}

/// In-place radix-2 FFT of a power-of-two length.
fn fft(a: &mut [(f32, f32)]) {
    let n = a.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (ang * k as f64).sin_cos();
                let (wr, wi) = (c as f32, s as f32);
                let (ur, ui) = a[start + k];
                let (xr, xi) = a[start + k + len / 2];
                let (vr, vi) = (xr * wr - xi * wi, xr * wi + xi * wr);
                a[start + k] = (ur + vr, ui + vi);
                a[start + k + len / 2] = (ur - vr, ui - vi);
            }
        }
        len <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_of_a_cosine_has_two_peaks() {
        let mut a: Vec<(f32, f32)> = (0..16).map(|i| ((2.0 * std::f32::consts::PI * 3.0 * i as f32 / 16.0).cos(), 0.0)).collect();
        fft(&mut a);
        let mag: Vec<f32> = a.iter().map(|(r, i)| (r * r + i * i).sqrt()).collect();
        assert!((mag[3] - 8.0).abs() < 1e-3 && (mag[13] - 8.0).abs() < 1e-3);
        assert!(mag.iter().enumerate().all(|(k, m)| k == 3 || k == 13 || *m < 1e-3));
    }

    #[test]
    fn fbank_frame_count_follows_kaldi_without_snip_edges() {
        assert_eq!(fbank(&vec![0.0; 16_000]).0, 100);
        assert_eq!(fbank(&vec![0.0; 16_079]).0, 100);
        assert_eq!(fbank(&vec![0.0; 16_080]).0, 101);
    }

    /// Writes Lana's fbank of a demo clip for comparison with lhotse's:
    /// cargo test -- --ignored dump_fbank
    #[test]
    #[ignore]
    fn dump_fbank() {
        let bytes = std::fs::read("../web/demo/de-vinson.wav").unwrap();
        let x = crate::local::audio::decode_16k_mono(bytes, "a.wav").unwrap();
        let (_, f) = fbank(&x);
        let raw: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(std::env::var("FBANK_OUT").unwrap(), raw).unwrap();
        let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(std::env::var("FBANK_OUT").unwrap() + ".pcm", raw).unwrap();
    }
}
