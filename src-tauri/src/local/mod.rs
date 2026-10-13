// SPDX-License-Identifier: GPL-3.0-or-later
// Recognition that runs inside Lana: whisper.cpp, no server, no key.
//
// It presents itself to the page as one more provider ("lana", kind "builtin"),
// answering the same progress / partial / cancel contract as the llama-swap
// services, so the page needs nothing special for it.

pub mod audio;
pub mod models;

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

use serde_json::{json, Value};
use whisper_rs::{DtwMode, DtwParameters, FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub use models::Models;

pub const PROVIDER_ID: &str = "lana";

#[derive(Default)]
struct Job {
    model: String,
    stage: &'static str,
    total: f64,
    started: Option<Instant>,
}

#[derive(Clone)]
pub struct Local {
    pub models: Models,
    job: Arc<Mutex<Option<Job>>>,
    /// Percent done, written from whisper's progress callback.
    percent: Arc<AtomicU64>,
    cancel: Arc<AtomicBool>,
    /// Rises once per job, as the llama-swap services' `job` does.
    serial: Arc<AtomicU64>,
}

impl Local {
    pub fn new(models: Models) -> Self {
        Local {
            models,
            job: Default::default(),
            percent: Default::default(),
            cancel: Default::default(),
            serial: Default::default(),
        }
    }

    /// The built-in provider, or None until a model has been downloaded.
    pub fn provider(&self) -> Option<Value> {
        let models: Vec<Value> = self
            .models
            .installed()
            .iter()
            .map(|e| json!({ "id": e.id, "role": "words", "label": format!("{} (on this computer)", e.label) }))
            .collect();
        (!models.is_empty()).then(|| {
            json!({
                "id": PROVIDER_ID,
                "label": "Built into Lana (this computer)",
                "kind": "builtin",
                "base": "",
                "control": "builtin",
                "models": models,
            })
        })
    }

    pub fn progress(&self) -> Value {
        let job = self.job.lock().unwrap();
        let serial = self.serial.load(Ordering::Relaxed);
        match job.as_ref() {
            Some(j) => {
                let pct = self.percent.load(Ordering::Relaxed) as f64 / 100.0;
                json!({
                    "active": true, "model": j.model, "stage": j.stage,
                    "done": (j.total * pct * 10.0).round() / 10.0, "total": (j.total * 10.0).round() / 10.0,
                    "elapsed": j.started.map_or(0, |s| s.elapsed().as_secs()),
                    "job": serial, "cues_ready": 0,
                })
            }
            None => json!({ "active": false, "job": serial, "cues_ready": 0 }),
        }
    }

    /// Words arrive all at once when the job ends: whisper.cpp's DTW word
    /// timing, which is what makes the boundaries usable for phonetics, cannot
    /// be combined with its per-segment callback.
    pub fn partial(&self, since: u64) -> Value {
        json!({ "cues": [], "since": since, "job": self.serial.load(Ordering::Relaxed) })
    }

    pub fn cancel(&self) -> Value {
        let active = self.job.lock().unwrap().is_some();
        if active {
            self.cancel.store(true, Ordering::Relaxed);
        }
        json!({ "ok": true, "cancelling": active })
    }

    /// Transcribe one upload. Blocking; call from spawn_blocking.
    pub fn transcribe(&self, model_id: &str, bytes: Vec<u8>, filename: &str, language: Option<&str>) -> (u16, Value) {
        let Some(entry) = models::find(model_id) else {
            return (404, json!({ "error": format!("Lana has no model called '{model_id}'.") }));
        };
        let path = self.models.path(&entry);
        if !path.is_file() {
            return (400, json!({ "error": format!("{} is not downloaded yet. Download it in Connection → Recognisers.", entry.label) }));
        }
        {
            let mut job = self.job.lock().unwrap();
            if let Some(j) = job.as_ref() {
                let secs = j.started.map_or(0, |s| s.elapsed().as_secs());
                return (429, json!({ "error": format!("busy: {} is running ({}, {secs} s). Press Stop, or wait.", j.model, j.stage) }));
            }
            *job = Some(Job { model: model_id.into(), stage: "reading audio", total: 0.0, started: Some(Instant::now()) });
        }
        self.serial.fetch_add(1, Ordering::Relaxed);
        self.percent.store(0, Ordering::Relaxed);
        self.cancel.store(false, Ordering::Relaxed);

        let result = self.run(&entry, &path, bytes, filename, language);
        *self.job.lock().unwrap() = None;
        match result {
            Ok(v) => (200, v),
            Err(_) if self.cancel.load(Ordering::Relaxed) => (499, json!({ "error": "stopped" })),
            Err(e) => (500, json!({ "error": e })),
        }
    }

    fn stage(&self, stage: &'static str) {
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.stage = stage;
        }
    }

    /// Load the model for one job. Not kept between jobs: loading takes
    /// 0.1-0.2 s, and holding up to 500 MB resident for occasional jobs costs
    /// more than it saves.
    fn context(&self, entry: &models::Entry, path: &std::path::Path) -> Result<WhisperContext, String> {
        let mut params = WhisperContextParameters::default();
        params.dtw_parameters = DtwParameters {
            mode: DtwMode::ModelPreset { model_preset: preset(entry.id) },
            ..Default::default()
        };
        WhisperContext::new_with_params(path.to_str().ok_or("model path is not UTF-8")?, params)
            .map_err(|e| format!("Could not load {}: {e}", entry.label))
    }

    fn run(&self, entry: &models::Entry, path: &std::path::Path, bytes: Vec<u8>, filename: &str, language: Option<&str>) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let duration = samples.len() as f64 / audio::WHISPER_RATE as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let ctx = self.context(entry, path)?;
        let mut state = ctx.create_state().map_err(|e| e.to_string())?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        // whisper wants a two-letter code or nothing. A three-letter code
        // (meant for MMS) would be refused, so fall back to detection.
        let lang = language.map(str::trim).filter(|l| l.len() == 2);
        params.set_language(Some(lang.unwrap_or("auto")));
        params.set_n_threads(threads());
        params.set_token_timestamps(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        let pct = self.percent.clone();
        params.set_progress_callback_safe(move |p: i32| pct.store(p.clamp(0, 100) as u64, Ordering::Relaxed));
        // Not set_abort_callback_safe: in whisper-rs 0.16 its trampoline reads
        // the boxed closure as the bare closure type, so every call returns
        // garbage. Linux happened to read "continue" on a first run and
        // "abort" later; Windows aborted the encoder at once (error -6). The
        // flag is all the callback needs, so hand whisper.cpp the flag itself.
        // It lives in self.cancel (an Arc), which outlives this full() call.
        unsafe {
            params.set_abort_callback(Some(should_abort));
            params.set_abort_callback_user_data(Arc::as_ptr(&self.cancel) as *mut std::ffi::c_void);
        }

        self.stage("recognising words");
        let t = Instant::now();
        state.full(params, &samples).map_err(|e| format!("whisper failed: {e}"))?;
        eprintln!("lana: {} transcribed {duration:.1} s of audio in {:.1} s", entry.id, t.elapsed().as_secs_f64());
        if self.cancel.load(Ordering::Relaxed) {
            return Err("stopped".into());
        }

        let eot = ctx.token_eot();
        let detected = state.full_lang_id_from_state();
        let mut cues = Vec::new();
        let mut text = String::new();
        for seg in state.as_iter() {
            let (a, b) = (seg.start_timestamp() as f64 / 100.0, seg.end_timestamp() as f64 / 100.0);
            let seg_text = seg.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default();
            text.push_str(&seg_text);
            // Tokens -> words: a token that starts with a space starts a word;
            // anything else (word pieces, punctuation) belongs to the last one.
            let mut words: Vec<(String, f64)> = Vec::new();
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else { continue };
                let data = tok.token_data();
                if data.id >= eot {
                    continue; // timestamps and control tokens
                }
                let piece = tok.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default();
                let t = if data.t_dtw >= 0 { data.t_dtw as f64 / 100.0 } else { data.t0 as f64 / 100.0 };
                if piece.starts_with(' ') || words.is_empty() {
                    words.push((piece.trim_start().to_owned(), t.clamp(a, b)));
                } else if let Some(last) = words.last_mut() {
                    last.0.push_str(&piece);
                }
            }
            // A word runs from its first token to the next word's first token.
            let n = words.len();
            let mut out = Vec::with_capacity(n);
            for k in 0..n {
                let start = words[k].1;
                let end = if k + 1 < n { words[k + 1].1 } else { b };
                let end = end.max(start + 0.02).min(b.max(start + 0.02));
                out.push(json!({ "text": words[k].0, "start": round3(start), "end": round3(end),
                                 "phones": [], "phone_times": [], "aligned": true }));
            }
            let mut cue = json!({ "start": round3(a), "end": round3(b), "words": out });
            if n == 0 {
                cue["plain"] = json!(seg_text.trim());
            }
            cues.push(cue);
        }
        Ok(json!({
            "text": text.trim(),
            "language": whisper_rs::get_lang_str(detected).unwrap_or(""),
            "duration": round3(duration),
            "cues": cues,
        }))
    }
}

/// ggml's abort callback: `user_data` is the job's cancel flag.
unsafe extern "C" fn should_abort(user_data: *mut std::ffi::c_void) -> bool {
    !user_data.is_null() && (*(user_data as *const AtomicBool)).load(Ordering::Relaxed)
}

fn preset(id: &str) -> whisper_rs::DtwModelPreset {
    models::find(id).map(|e| e.dtw).unwrap_or(whisper_rs::DtwModelPreset::Base)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Leave a core for the interface and Praat.
fn threads() -> i32 {
    std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(1).clamp(1, 8)) as i32
}

// ---------------------------------------------------------------- routes

use axum::{
    extract::{Path as UrlPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

/// The settings screen's model manager. Desktop only, like the rest of settings.
pub fn router(local: Local) -> Router {
    Router::new()
        .route("/api/local/models", get(list_models))
        .route("/api/local/models/{id}", post(download_model).delete(delete_model))
        .with_state(local)
}

async fn list_models(State(l): State<Local>) -> Json<Value> {
    Json(l.models.list())
}

async fn download_model(State(l): State<Local>, UrlPath(id): UrlPath<String>) -> Response {
    match l.models.start(&id) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => (StatusCode::NOT_FOUND, Json(json!({ "error": e }))).into_response(),
    }
}

async fn delete_model(State(l): State<Local>, UrlPath(id): UrlPath<String>) -> Response {
    if l.job.lock().unwrap().as_ref().is_some_and(|j| j.model == id) {
        return (StatusCode::CONFLICT, Json(json!({ "error": "That model is transcribing right now." }))).into_response();
    }
    match l.models.delete(&id) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}
