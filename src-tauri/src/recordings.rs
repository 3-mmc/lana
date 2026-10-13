// SPDX-License-Identifier: GPL-3.0-or-later
// Live recordings, written to disk while they happen.
//
// Fieldwork does not get a second take. A recording held in memory until Stop is
// lost to a crash, a closed lid or a flat battery, so the page streams PCM here
// in small chunks and every chunk is appended to the WAV at once. The header's
// two size fields are rewritten after each chunk, so the file on disk is a
// valid, complete WAV at every moment, not just after Stop.
//
// A metadata sidecar (<name>.json) sits beside each recording. Its `status`
// reads "recording" until Stop, which is how an interrupted session can be told
// apart from a finished one later.

use std::{
    collections::HashMap,
    fs::File,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

struct Session {
    file: File,
    name: String,
    data_bytes: u64,
    rate: u32,
    channels: u16,
    bits: u16,
    meta: Value,
}

#[derive(Clone)]
pub struct Recordings {
    dir: PathBuf,
    live: Arc<Mutex<HashMap<String, Session>>>,
}

pub fn router(dir: PathBuf) -> Router {
    Router::new()
        .route("/api/record/start", post(start))
        .route("/api/record/{id}/chunk", post(chunk))
        .route("/api/record/{id}/finish", post(finish))
        .route("/api/recordings/{name}", get(serve))
        .with_state(Recordings { dir, live: Default::default() })
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// RIFF/WAVE header for integer PCM. Sizes are filled from `data_bytes`.
fn wav_header(rate: u32, channels: u16, bits: u16, data_bytes: u64) -> Vec<u8> {
    let data = data_bytes.min(u32::MAX as u64 - 36) as u32;
    let block = channels * bits / 8;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // integer PCM
    h.extend_from_slice(&channels.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * block as u32).to_le_bytes());
    h.extend_from_slice(&block.to_le_bytes());
    h.extend_from_slice(&bits.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data.to_le_bytes());
    h
}

/// A filename-safe version of what the user typed as the speaker code.
fn slug(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' { c } else { '_' })
        .collect();
    s.trim_matches('_').chars().take(40).collect()
}

impl Recordings {
    fn write_sidecar(&self, s: &Session, status: &str) {
        let mut meta = s.meta.clone();
        if let Some(o) = meta.as_object_mut() {
            o.insert("status".into(), json!(status));
            o.insert("file".into(), json!(s.name));
            o.insert("sample_rate".into(), json!(s.rate));
            o.insert("channels".into(), json!(s.channels));
            o.insert("bits".into(), json!(s.bits));
            o.insert("duration_s".into(), json!(s.data_bytes as f64 / (s.rate as f64 * (s.channels * s.bits / 8) as f64)));
            o.insert("recorded_with".into(), json!(concat!("Lana ", env!("CARGO_PKG_VERSION"))));
        }
        let path = self.dir.join(&s.name).with_extension("json");
        let _ = std::fs::write(path, serde_json::to_string_pretty(&meta).unwrap() + "\n");
    }
}

/// `{rate, channels, bits, meta: {speaker, language, place, consent, notes}}`
async fn start(State(r): State<Recordings>, Json(body): Json<Value>) -> Response {
    let rate = body.get("rate").and_then(Value::as_u64).unwrap_or(0) as u32;
    let channels = body.get("channels").and_then(Value::as_u64).unwrap_or(1) as u16;
    let bits = body.get("bits").and_then(Value::as_u64).unwrap_or(24) as u16;
    if !(8_000..=384_000).contains(&rate) || !(1..=2).contains(&channels) || !matches!(bits, 16 | 24) {
        return err(StatusCode::BAD_REQUEST, "unsupported recording format");
    }
    if let Err(e) = std::fs::create_dir_all(&r.dir) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Cannot create {}: {e}", r.dir.display()));
    }
    let now = chrono::Local::now();
    let mut meta = body.get("meta").cloned().unwrap_or_else(|| json!({}));
    if !meta.is_object() {
        meta = json!({});
    }
    meta["started"] = json!(now.to_rfc3339());
    let speaker = slug(meta.get("speaker").and_then(Value::as_str).unwrap_or(""));
    let stem = format!("{}{}", now.format("%Y-%m-%d_%H%M%S"), if speaker.is_empty() { String::new() } else { format!("_{speaker}") });
    // Never overwrite: a second recording in the same second gets a suffix.
    let mut name = format!("{stem}.wav");
    for n in 2.. {
        if !r.dir.join(&name).exists() {
            break;
        }
        name = format!("{stem}_{n}.wav");
    }
    let path = r.dir.join(&name);
    let mut file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Cannot create {}: {e}", path.display())),
    };
    if let Err(e) = file.write_all(&wav_header(rate, channels, bits, 0)) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    let session = Session { file, name: name.clone(), data_bytes: 0, rate, channels, bits, meta };
    r.write_sidecar(&session, "recording");
    let id = name.trim_end_matches(".wav").to_owned();
    r.live.lock().unwrap().insert(id.clone(), session);
    Json(json!({ "id": id, "file": name, "path": path.display().to_string() })).into_response()
}

/// Raw little-endian PCM in the format given at start.
async fn chunk(State(r): State<Recordings>, Path(id): Path<String>, pcm: Bytes) -> Response {
    let mut live = r.live.lock().unwrap();
    let Some(s) = live.get_mut(&id) else {
        return err(StatusCode::NOT_FOUND, "no such recording in progress");
    };
    let block = (s.channels * s.bits / 8) as usize;
    if pcm.len() % block != 0 {
        return err(StatusCode::BAD_REQUEST, "chunk is not whole sample frames");
    }
    let result = (|| -> std::io::Result<()> {
        s.file.seek(SeekFrom::End(0))?;
        s.file.write_all(&pcm)?;
        s.data_bytes += pcm.len() as u64;
        // Keep the header true after every chunk: the file is complete at all times.
        let h = wav_header(s.rate, s.channels, s.bits, s.data_bytes);
        s.file.seek(SeekFrom::Start(4))?;
        s.file.write_all(&h[4..8])?;
        s.file.seek(SeekFrom::Start(40))?;
        s.file.write_all(&h[40..44])?;
        Ok(())
    })();
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) if e.raw_os_error() == Some(28) || e.raw_os_error() == Some(112) => {
            err(StatusCode::INSUFFICIENT_STORAGE, "The disk is full. What was recorded so far is saved.")
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not write the recording: {e}")),
    }
}

async fn finish(State(r): State<Recordings>, Path(id): Path<String>) -> Response {
    let Some(mut s) = r.live.lock().unwrap().remove(&id) else {
        return err(StatusCode::NOT_FOUND, "no such recording in progress");
    };
    let _ = s.file.flush();
    let _ = s.file.sync_all();
    r.write_sidecar(&s, "complete");
    let secs = s.data_bytes as f64 / (s.rate as f64 * (s.channels * s.bits / 8) as f64);
    Json(json!({
        "file": s.name, "path": r.dir.join(&s.name).display().to_string(),
        "duration": secs, "bytes": s.data_bytes + 44,
    }))
    .into_response()
}

/// A finished recording, by file name, for the page to open. Only plain names
/// inside the recordings folder: no separators, no parent references.
async fn serve(State(r): State<Recordings>, Path(name): Path<String>) -> Response {
    if name.contains(['/', '\\']) || name.starts_with('.') || !name.ends_with(".wav") {
        return err(StatusCode::BAD_REQUEST, "bad name");
    }
    match tokio::fs::read(r.dir.join(&name)).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, "audio/wav")], bytes).into_response(),
        Err(e) => err(StatusCode::NOT_FOUND, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_44_bytes_and_sizes_add_up() {
        let h = wav_header(48_000, 1, 24, 300);
        assert_eq!(h.len(), 44);
        assert_eq!(u32::from_le_bytes(h[4..8].try_into().unwrap()), 336);
        assert_eq!(u32::from_le_bytes(h[40..44].try_into().unwrap()), 300);
        assert_eq!(u32::from_le_bytes(h[28..32].try_into().unwrap()), 144_000); // byte rate
        assert_eq!(u16::from_le_bytes(h[32..34].try_into().unwrap()), 3); // block align
    }

    #[test]
    fn slug_keeps_letters_and_drops_paths() {
        assert_eq!(slug("../AV 03/x"), "AV_03_x");
        assert_eq!(slug("Хадижат"), "Хадижат");
    }
}
