// SPDX-License-Identifier: GPL-3.0-or-later
// Frame probabilities from the CTC models, which sherpa-onnx computed but
// never returned: its result was tokens and times only. Confidence, a phone's
// runner-up readings, re-timing corrected labels and search by example all
// need the probability of every symbol in every frame, so Lana runs these
// models itself (through runtime.rs, like every model but Whisper).

use std::{path::Path, sync::{Mutex, OnceLock}};

use ort::value::Tensor;

use super::runtime::{Model, Role};

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

    /// Frames `from..to`, clamped to what there is.
    pub fn slice(&self, from: usize, to: usize) -> Mat {
        let (from, to) = (from.min(self.t), to.min(self.t).max(from.min(self.t)));
        Mat { t: to - from, v: self.v, lp: self.lp[from * self.v..to * self.v].to_vec() }
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
    model: Option<Mutex<Model>>,
    native: Option<Mutex<super::mac_models::Worker>>,
    input: Input,
}

impl Net {
    pub fn load(model: &Path, input: Input, threads: i32) -> Result<Self, String> {
        if super::mac_models::model_id(model).is_some() {
            return Ok(Net { model: None, native: Some(Mutex::new(super::mac_models::Worker::load(model)?)), input });
        }
        Ok(Net { model: Some(Mutex::new(Model::load(model, Role::Heavy, threads)?)), native: None, input })
    }

    pub fn native_tokens(&self) -> Option<Vec<String>> {
        self.native.as_ref().and_then(|n| n.lock().unwrap().tokens.clone())
    }

    /// Log-probabilities for one piece of 16 kHz mono audio.
    pub fn log_probs(&self, samples: &[f32]) -> Result<Mat, String> {
        if let Some(native) = &self.native {
            if samples.len() < 400 { return Ok(Mat::default()); }
            let mut mat = native.lock().unwrap().logits(&normalised(samples))?;
            for row in mat.lp.chunks_mut(mat.v) { log_softmax(row); }
            return Ok(mat);
        }
        let err = |e: ort::Error| format!("the model failed: {e}");
        let mut model = self.model.as_ref().unwrap().lock().unwrap();
        let mut m = match self.input {
            Input::Fbank => {
                let (frames, feats) = fbank(samples);
                if frames == 0 {
                    return Ok(Mat::default());
                }
                model.with(|session| {
                    let x = Tensor::from_array(([1usize, frames, MEL_BINS], feats.clone())).map_err(err)?;
                    let lens = Tensor::from_array(([1usize], vec![frames as i64])).map_err(err)?;
                    let out = session.run(ort::inputs!["x" => x, "x_lens" => lens]).map_err(err)?;
                    let (shape, lp) = out["log_probs"].try_extract_tensor::<f32>().map_err(err)?;
                    let (_, len) = out["log_probs_len"].try_extract_tensor::<i64>().map_err(err)?;
                    let (t, v) = (len[0].max(0) as usize, shape[2] as usize);
                    Ok(Mat { t, v, lp: lp[..t * v].to_vec() })
                })?
            }
            Input::Samples => {
                if samples.len() < 400 {
                    return Ok(Mat::default());
                }
                let x = normalised(samples);
                model.with(|session| {
                    let x = Tensor::from_array(([1usize, x.len()], x.clone())).map_err(err)?;
                    let out = session.run(ort::inputs!["x" => x]).map_err(err)?;
                    let (shape, logits) = out[0].try_extract_tensor::<f32>().map_err(err)?;
                    let (t, v) = (shape[1] as usize, shape[2] as usize);
                    Ok(Mat { t, v, lp: logits.to_vec() })
                })?
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
    static KALDI: OnceLock<Fbank> = OnceLock::new();
    KALDI.get_or_init(|| Fbank::kaldi(MEL_BINS)).compute(x)
}

/// A log mel filterbank on Kaldi's framing (see `fbank`), with either
/// Kaldi's mel bands or librosa's.
pub struct Fbank {
    pub bins: usize,
    remove_dc: bool,
    window: Vec<f32>,
    banks: Vec<(usize, Vec<f32>)>,
}

impl Fbank {
    /// Kaldi's bands, 20 Hz to 7600 Hz, triangles on the mel scale.
    pub fn kaldi(bins: usize) -> Self {
        Fbank { bins, remove_dc: true, window: povey(), banks: kaldi_banks(bins) }
    }

    /// NeMo's features as sherpa-onnx computes them for Parakeet: librosa's
    /// bands (Slaney's mel scale, triangles in Hz, area-normalised) from 0 Hz
    /// to 7600 Hz, the DC offset left in.
    pub fn librosa(bins: usize) -> Self {
        Fbank { bins, remove_dc: false, window: povey(), banks: librosa_banks(bins) }
    }

    pub fn compute(&self, x: &[f32]) -> (usize, Vec<f32>) {
        let n = x.len();
        let frames = (n + SHIFT / 2) / SHIFT;
        let mut out = Vec::with_capacity(frames * self.bins);
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
            if self.remove_dc {
                let mean = w.iter().sum::<f32>() / LENGTH as f32;
                for v in w.iter_mut() {
                    *v -= mean;
                }
            }
            for i in (1..LENGTH).rev() {
                w[i] -= 0.97 * w[i - 1];
            }
            w[0] -= 0.97 * w[0];
            for (i, b) in buf.iter_mut().enumerate() {
                *b = (if i < LENGTH { w[i] * self.window[i] } else { 0.0 }, 0.0);
            }
            fft(&mut buf);
            for (first, weights) in &self.banks {
                let e: f32 = weights.iter().enumerate().map(|(k, wt)| {
                    let (re, im) = buf[first + k];
                    wt * (re * re + im * im)
                }).sum();
                out.push(e.max(f32::EPSILON).ln());
            }
        }
        (frames, out)
    }
}

fn povey() -> Vec<f32> {
    (0..LENGTH)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / (LENGTH - 1) as f64;
            (0.5 - 0.5 * a.cos()).powf(0.85) as f32
        })
        .collect()
}

/// Kaldi's mel filterbank: each band as its first FFT bin and its weights.
fn kaldi_banks(bins: usize) -> Vec<(usize, Vec<f32>)> {
    let mel = |f: f64| 1127.0 * (1.0 + f / 700.0).ln();
    let (lo, hi) = (mel(20.0), mel(8000.0 - 400.0));
    let delta = (hi - lo) / (bins + 1) as f64;
    let bin_hz = 16_000.0 / FFT as f64;
    (0..bins)
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

/// librosa's mel filterbank as kaldi-native-fbank builds it (is_librosa):
/// band edges evenly spaced on Slaney's mel scale, triangles drawn in Hz
/// over the FFT bins including Nyquist, each scaled to unit area. In single
/// precision, as there.
fn librosa_banks(bins: usize) -> Vec<(usize, Vec<f32>)> {
    let mel = |f: f32| if f <= 1000.0 { f * 3.0 / 200.0 } else { 15.0 + 14.545_079 * (f / 1000.0).ln() };
    let hz = |m: f32| if m <= 15.0 { 200.0 / 3.0 * m } else { 1000.0 * ((m - 15.0) * 0.068_751_78).exp() };
    let (lo, hi) = (mel(0.0), mel(8000.0 - 400.0));
    let delta = (hi - lo) / (bins + 1) as f32;
    let bin_hz = 16_000.0 / FFT as f32;
    (0..bins)
        .map(|b| {
            let (l, c, r) = (hz(lo + b as f32 * delta), hz(lo + (b + 1) as f32 * delta), hz(lo + (b + 2) as f32 * delta));
            let mut first = None;
            let mut weights = Vec::new();
            for i in 0..=FFT / 2 {
                let f = bin_hz * i as f32;
                if f > l && f < r {
                    first.get_or_insert(i);
                    let w = if f <= c { (f - l) / (c - l) } else { (r - f) / (r - c) };
                    weights.push(w * 2.0 / (r - l));
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

    /// Writes Lana's fbank of a demo clip (FBANK_CLIP, de-vinson by default)
    /// for comparison with lhotse's or kaldi-native-fbank's; NeMo's bands go
    /// to <FBANK_OUT>.nemo: cargo test -- --ignored dump_fbank
    #[test]
    #[ignore]
    fn dump_fbank() {
        let clip = std::env::var("FBANK_CLIP").unwrap_or("de-vinson".into());
        let bytes = std::fs::read(format!("../web/demo/{clip}.wav")).unwrap();
        let x = crate::local::audio::decode_16k_mono(bytes, "a.wav").unwrap();
        let (_, f) = fbank(&x);
        let raw: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(std::env::var("FBANK_OUT").unwrap(), raw).unwrap();
        let (_, f) = Fbank::librosa(128).compute(&x);
        let raw: Vec<u8> = f.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(std::env::var("FBANK_OUT").unwrap() + ".nemo", raw).unwrap();
        let raw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(std::env::var("FBANK_OUT").unwrap() + ".pcm", raw).unwrap();
    }
}

