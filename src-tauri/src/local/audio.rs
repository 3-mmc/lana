// SPDX-License-Identifier: GPL-3.0-or-later
// Any common audio file -> 16 kHz mono f32, which is what whisper.cpp reads.

use std::io::Cursor;

use symphonia::core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    errors::Error,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

pub const WHISPER_RATE: u32 = 16_000;

/// Decode, mix to mono and resample. The error is a sentence for the user.
pub fn decode_16k_mono(bytes: Vec<u8>, filename: &str) -> Result<Vec<f32>, String> {
    let (mono, rate) = decode_mono(bytes, filename)?;
    Ok(resample(&mono, rate, WHISPER_RATE))
}

fn decode_mono(bytes: Vec<u8>, filename: &str) -> Result<(Vec<f32>, u32), String> {
    let unreadable = |e: Error| format!("Lana cannot read this audio file ({e}).");
    let mut hint = Hint::new();
    if let Some(ext) = std::path::Path::new(filename).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(unreadable)?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| "This file has no audio track.".to_owned())?;
    let track_id = track.id;
    let mut rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(unreadable)?;

    let mut mono = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(Error::ResetRequired) => break,
            Err(e) => return Err(unreadable(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A damaged packet costs a few milliseconds of audio, not the file.
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(unreadable(e)),
        };
        let spec = *decoded.spec();
        rate = spec.rate;
        let channels = spec.channels.count().max(1);
        let sb = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if sb.capacity() < decoded.capacity() * channels {
            *sb = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        sb.copy_interleaved_ref(decoded);
        for frame in sb.samples().chunks(channels) {
            mono.push(frame.iter().sum::<f32>() / channels as f32);
        }
    }
    if mono.is_empty() || rate == 0 {
        return Err("This file decoded to no audio.".into());
    }
    Ok((mono, rate))
}

/// Windowed-sinc low-pass, then linear interpolation. Not studio quality, but
/// the low-pass is what matters: without it everything above 8 kHz folds back
/// into the band whisper listens to.
fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to {
        return x.to_vec();
    }
    let filtered = if to < from { lowpass(x, 0.45 * to as f64 / from as f64) } else { x.to_vec() };
    let step = from as f64 / to as f64;
    let n = ((x.len() as f64) / step).floor() as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * step;
            let k = pos as usize;
            let frac = (pos - k as f64) as f32;
            let a = filtered[k];
            let b = *filtered.get(k + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

/// `cutoff` is a fraction of the input rate (0..0.5).
fn lowpass(x: &[f32], cutoff: f64) -> Vec<f32> {
    const HALF: isize = 48;
    let taps: Vec<f32> = (-HALF..=HALF)
        .map(|n| {
            let n = n as f64;
            let sinc = if n == 0.0 { 2.0 * cutoff } else { (2.0 * std::f64::consts::PI * cutoff * n).sin() / (std::f64::consts::PI * n) };
            // Blackman window
            let w = 0.42 + 0.5 * (std::f64::consts::PI * n / HALF as f64).cos() + 0.08 * (2.0 * std::f64::consts::PI * n / HALF as f64).cos();
            (sinc * w) as f32
        })
        .collect();
    let len = x.len() as isize;
    (0..len)
        .map(|i| {
            let mut acc = 0.0f32;
            for (j, t) in taps.iter().enumerate() {
                let k = i + j as isize - HALF;
                if (0..len).contains(&k) {
                    acc += x[k as usize] * t;
                }
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_duration_and_a_low_tone() {
        let from = 48_000;
        let x: Vec<f32> = (0..from).map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin()).collect();
        let y = resample(&x, from, WHISPER_RATE);
        assert_eq!(y.len(), WHISPER_RATE as usize);
        let rms = (y[1000..15000].iter().map(|v| v * v).sum::<f32>() / 14000.0).sqrt();
        assert!((rms - 0.707).abs() < 0.03, "440 Hz should pass, rms {rms}");
    }

    #[test]
    fn resample_removes_what_would_alias() {
        let from = 48_000;
        let x: Vec<f32> = (0..from).map(|i| (2.0 * std::f32::consts::PI * 12_000.0 * i as f32 / from as f32).sin()).collect();
        let y = resample(&x, from, WHISPER_RATE);
        let rms = (y[1000..15000].iter().map(|v| v * v).sum::<f32>() / 14000.0).sqrt();
        assert!(rms < 0.02, "12 kHz must not fold back into the 16 kHz signal, rms {rms}");
    }
}
