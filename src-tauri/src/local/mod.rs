// SPDX-License-Identifier: GPL-3.0-or-later
// Recognition that runs inside Lana, with no server and no key: whisper.cpp,
// Omnilingual ASR and Parakeet for words, ZIPA for IPA phones, pyannote with
// CAM++ for speakers, an AudioSet tagger for other sounds, and Silero VAD to
// cut the audio into pieces at pauses; everything but whisper.cpp through
// sherpa-onnx.
//
// It presents itself to the page as one more provider ("lana", kind "builtin"),
// answering the same progress / partial / cancel / recluster contract as the
// llama-swap services, so the page needs nothing special for it.

pub mod audio;
pub mod models;
pub mod ctc;
pub mod mms;
pub mod parakeet;
pub mod speakers;
pub mod tags;
pub mod vad;

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
use models::Kind;

/// One transcription request, as the broker received it.
pub struct Request {
    pub model: String,
    pub bytes: Vec<u8>,
    pub filename: String,
    pub language: Option<String>,
    /// Speaker models only: an exact count, and the clustering threshold.
    pub speakers: Option<i32>,
    pub threshold: Option<f32>,
}

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
    /// Cues the running job has finished so far, for /partial: each 30 s
    /// piece of the recording adds its own as soon as it is read.
    done_cues: Arc<Mutex<Vec<Value>>>,
    /// A speaker pass runs beside the recogniser, not after it, as the
    /// diarizer service does; it has its own slot and no progress feed.
    side: Arc<AtomicBool>,
    /// The last speaker pass, kept so the speaker count can be changed
    /// without reading the audio again.
    held: Arc<Mutex<Option<speakers::Held>>>,
}

impl Local {
    pub fn new(models: Models) -> Self {
        vad::use_model(models.vad_path());
        Local {
            models,
            job: Default::default(),
            percent: Default::default(),
            cancel: Default::default(),
            serial: Default::default(),
            done_cues: Default::default(),
            side: Default::default(),
            held: Default::default(),
        }
    }

    /// The built-in provider, or None until a model has been downloaded.
    pub fn provider(&self) -> Option<Value> {
        let mut models: Vec<Value> = self
            .models
            .installed()
            .iter()
            .map(|e| json!({ "id": e.id, "role": e.kind.role(), "label": format!("{} (on this computer)", e.label) }))
            .collect();
        // One word model per MMS language on this computer.
        for code in self.models.mms_installed() {
            let name = mms::name_of(&code).unwrap_or(&code);
            models.push(json!({ "id": format!("mms-{code}"), "role": "words",
                                "label": format!("MMS · {name} (on this computer, non-commercial)") }));
        }
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
                    "job": serial, "cues_ready": self.done_cues.lock().unwrap().len(),
                })
            }
            None => json!({ "active": false, "job": serial, "cues_ready": self.done_cues.lock().unwrap().len() }),
        }
    }

    /// The cues finished since `since`, in the shape the llama-swap services
    /// answer: every model here reads the recording in pieces of up to 30 s and
    /// adds each piece's cues as it finishes, so a long recording shows its
    /// first minute while the rest is still being read.
    pub fn partial(&self, since: u64) -> Value {
        let cues = self.done_cues.lock().unwrap();
        let from = (since as usize).min(cues.len());
        let model = self.job.lock().unwrap().as_ref().map(|j| j.model.clone());
        json!({
            "cues": &cues[from..], "since": from, "total": cues.len(),
            "job": self.serial.load(Ordering::Relaxed), "model": model,
            "complete": model.is_none(),
        })
    }

    /// Whether a model is downloading or running; the models folder must not
    /// move under either.
    pub fn busy(&self) -> bool {
        self.job.lock().unwrap().is_some()
            || self.side.load(Ordering::Relaxed)
            || self.models.downloads.lock().unwrap().values().any(|d| d.error.is_none())
    }

    pub fn cancel(&self) -> Value {
        let active = self.job.lock().unwrap().is_some();
        if active {
            self.cancel.store(true, Ordering::Relaxed);
        }
        json!({ "ok": true, "cancelling": active })
    }

    /// Transcribe one upload. Blocking; call from spawn_blocking.
    pub fn transcribe(&self, req: Request) -> (u16, Value) {
        let model_id = req.model.as_str();
        if let Some(code) = model_id.strip_prefix("mms-") {
            if !self.models.mms_lang_installed(code) {
                return (400, json!({ "error": format!("MMS for '{code}' is not downloaded. Add the language in Connection → Recognisers.") }));
            }
            let dir = self.models.mms_lang_dir(code);
            return self.with_job(model_id, |me| me.run_ctc_at(ctc::Engine::Omnilingual, &dir.join("model.onnx"), &dir.join("tokens.txt"), model_id, req.bytes, &req.filename));
        }
        let Some(entry) = models::find(model_id) else {
            return (404, json!({ "error": format!("Lana has no model called '{model_id}'.") }));
        };
        if !self.models.is_installed(&entry) {
            return (400, json!({ "error": format!("{} is not downloaded yet. Download it in Connection → Recognisers.", entry.label) }));
        }
        if entry.kind == Kind::Speakers {
            return self.diarize(&entry, req);
        }
        if entry.kind == Kind::Whisper {
            if let Err(e) = whisper_supported() {
                return (400, json!({ "error": e }));
            }
        }
        let path = self.models.path(&entry);
        self.with_job(model_id, |me| match entry.kind {
            Kind::Phones | Kind::Letters => me.run_ctc(&entry, req.bytes, &req.filename),
            Kind::Transducer => me.run_parakeet(&entry, req.bytes, &req.filename),
            Kind::Tags => me.run_tags(&entry, req.bytes, &req.filename),
            _ => me.run(&entry, &path, req.bytes, &req.filename, req.language.as_deref()),
        })
    }

    /// Run one recogniser job in the single job slot that progress, partial
    /// and cancel report on.
    fn with_job(&self, model_id: &str, f: impl FnOnce(&Self) -> Result<Value, String>) -> (u16, Value) {
        {
            let mut job = self.job.lock().unwrap();
            if let Some(j) = job.as_ref() {
                let secs = j.started.map_or(0, |s| s.elapsed().as_secs());
                return (429, json!({ "error": format!("busy: {} is running ({}, {secs} s). Press Stop, or wait.", j.model, j.stage) }));
            }
            *job = Some(Job { model: model_id.into(), stage: "reading audio", total: 0.0, started: Some(Instant::now()) });
        }
        self.done_cues.lock().unwrap().clear();
        self.serial.fetch_add(1, Ordering::Relaxed);
        self.percent.store(0, Ordering::Relaxed);
        self.cancel.store(false, Ordering::Relaxed);
        let result = f(self);
        *self.job.lock().unwrap() = None;
        match result {
            Ok(v) => (200, v),
            Err(_) if self.cancel.load(Ordering::Relaxed) => (499, json!({ "error": "stopped" })),
            Err(e) => (500, json!({ "error": e })),
        }
    }

    fn run_ctc(&self, entry: &models::Entry, bytes: Vec<u8>, filename: &str) -> Result<Value, String> {
        let engine = if entry.kind == Kind::Letters { ctc::Engine::Omnilingual } else { ctc::Engine::Zipa };
        let (model, tokens) = (self.models.named(entry, "model.int8.onnx"), self.models.named(entry, "tokens.txt"));
        self.run_ctc_at(engine, &model, &tokens, entry.id, bytes, filename)
    }

    /// A CTC model given by its files. MMS runs here too: it has the same
    /// shape as Omnilingual (raw audio in, characters with a word separator out).
    fn run_ctc_at(&self, engine: ctc::Engine, model: &std::path::Path, tokens: &std::path::Path, id: &str, bytes: Vec<u8>, filename: &str) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let duration = samples.len() as f64 / audio::WHISPER_RATE as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let rec = ctc::Ctc::load(engine, model, tokens, threads())?;
        self.stage(if engine == ctc::Engine::Zipa { "reading phones" } else { "recognising words" });
        let t = Instant::now();
        let (pct, cancel, done_cues) = (self.percent.clone(), self.cancel.clone(), self.done_cues.clone());
        let found = rec.recognise(&samples, audio::WHISPER_RATE, |done, piece| {
            pct.store((done / duration.max(1e-3) * 100.0).clamp(0.0, 100.0) as u64, Ordering::Relaxed);
            let cues = if engine == ctc::Engine::Omnilingual { ctc::word_cues(piece, duration) } else { ctc::cues(piece, duration) };
            done_cues.lock().unwrap().extend(cues);
            !cancel.load(Ordering::Relaxed)
        })?;
        eprintln!("lana: {id} read {} symbols in {duration:.1} s of audio in {:.1} s", found.len(), t.elapsed().as_secs_f64());
        if engine == ctc::Engine::Omnilingual {
            let cues = ctc::word_cues(&found, duration);
            let text = found.iter().map(|(c, _)| c.as_str()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
            return Ok(json!({ "text": text, "duration": round3(duration), "cues": cues }));
        }
        let cues = ctc::cues(&found, duration);
        let text = cues
            .iter()
            .map(|c| c["words"][0]["phones"].as_array().map(|p| p.iter().filter_map(Value::as_str).collect::<String>()).unwrap_or_default())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(json!({ "text": text, "duration": round3(duration), "cues": cues }))
    }

    /// Parakeet over the recording in VAD pieces of up to 30 s, each piece's
    /// words on the page as it finishes. Pieces VAD found no speech in are
    /// skipped, as they are for Whisper: a transducer invents less in
    /// silence, but reading it is still time spent for nothing. The language
    /// is the model's own guess; it cannot be told one.
    fn run_parakeet(&self, entry: &models::Entry, bytes: Vec<u8>, filename: &str) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let rate = audio::WHISPER_RATE;
        let duration = samples.len() as f64 / rate as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let dir = self.models.path(entry);
        let rec = parakeet::Parakeet::load(dir.parent().ok_or("no model folder")?, threads())?;
        self.stage("recognising words");
        let t = Instant::now();
        let mut words = Vec::new();
        for p in vad::plan(&samples, rate, 30.0) {
            let (offset, end) = (p.range.start as f64 / rate as f64, p.range.end as f64 / rate as f64);
            if p.speech {
                let piece = rec.words(&samples[p.range.clone()], rate, offset, end);
                self.done_cues.lock().unwrap().extend(parakeet::cues(&piece));
                words.extend(piece);
            }
            self.percent.store((end / duration.max(1e-3) * 100.0).clamp(0.0, 100.0) as u64, Ordering::Relaxed);
            if self.cancel.load(Ordering::Relaxed) {
                return Err("stopped".into());
            }
        }
        eprintln!("lana: {} read {} words in {duration:.1} s of audio in {:.1} s", entry.id, words.len(), t.elapsed().as_secs_f64());
        let text = words.iter().map(|w| w.0.as_str()).collect::<Vec<_>>().join(" ");
        Ok(json!({ "text": text, "duration": round3(duration), "cues": parakeet::cues(&words) }))
    }

    /// The sound tagger, one window per second, each second's sound on the
    /// page as soon as its run of hops is closed.
    fn run_tags(&self, entry: &models::Entry, bytes: Vec<u8>, filename: &str) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let rate = audio::WHISPER_RATE;
        let duration = samples.len() as f64 / rate as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let tagger = tags::Tagger::load(&self.models.named(entry, "model.int8.onnx"), &self.models.named(entry, "labels.csv"), threads())?;
        self.stage("listening for sounds");
        let t = Instant::now();
        let (pct, cancel) = (self.percent.clone(), self.cancel.clone());
        let hops = tagger.label(&samples, rate, |done| {
            pct.store((done / duration.max(1e-3) * 100.0).clamp(0.0, 100.0) as u64, Ordering::Relaxed);
            !cancel.load(Ordering::Relaxed)
        })?;
        let events = tags::events(&hops, duration);
        eprintln!("lana: {} found {} sounds in {duration:.1} s of audio in {:.1} s", entry.id, events.len(), t.elapsed().as_secs_f64());
        let cues = tags::cues(&events);
        self.done_cues.lock().unwrap().extend(cues.iter().cloned());
        let text = events.iter().map(|e| e.0.as_str()).collect::<Vec<_>>().join(", ");
        Ok(json!({ "text": text, "duration": round3(duration), "cues": cues }))
    }

    fn speaker_models(&self, entry: &models::Entry) -> speakers::Models {
        speakers::models_at(&self.models.named(entry, "segmentation.onnx"), &self.models.named(entry, "campplus.onnx"), threads())
    }

    /// One speaker pass, in its own slot. Answers as the diarizer service
    /// does, hash included, so the page can re-cluster it afterwards.
    fn diarize(&self, entry: &models::Entry, req: Request) -> (u16, Value) {
        if self.side.swap(true, Ordering::SeqCst) {
            return (429, json!({ "error": "busy: the speaker model is already running. Wait for it to finish." }));
        }
        let result = (|| -> Result<Value, String> {
            use sha2::{Digest, Sha256};
            let hash: String = Sha256::digest(&req.bytes)[..16].iter().map(|b| format!("{b:02x}")).collect();
            let samples = audio::decode_16k_mono(req.bytes, &req.filename)?;
            let duration = samples.len() as f64 / speakers::RATE as f64;
            if duration < 0.5 {
                return Err("The recording is too short to tell voices apart.".into());
            }
            let t = Instant::now();
            let n = req.speakers.unwrap_or(0);
            let turns = speakers::diarize(&self.speaker_models(entry), &samples, n, req.threshold.unwrap_or(0.7))?;
            let found = speakers::count(&turns);
            eprintln!("lana: speakers: {duration:.1} s of audio, {found} voices, {} turns, {:.1} s", turns.len(), t.elapsed().as_secs_f64());
            let cues = speakers::cues(&turns);
            *self.held.lock().unwrap() = Some(speakers::Held { hash: hash.clone(), samples, turns: turns.clone(), windows: None });
            Ok(json!({
                "model": entry.id, "embedding": "campplus", "duration": round3(duration),
                "speakers": found, "turns": turns.len(), "cues": cues, "hash": hash,
            }))
        })();
        self.side.store(false, Ordering::SeqCst);
        match result {
            Ok(v) => (200, v),
            Err(e) => (500, json!({ "error": e })),
        }
    }

    /// Change the number of voices without reading the audio again.
    pub fn recluster(&self, hash: &str, speakers: usize) -> (u16, Value) {
        let Some(entry) = models::catalogue().into_iter().find(|e| e.kind == Kind::Speakers) else {
            return (404, json!({ "error": "no speaker model" }));
        };
        if speakers < 1 {
            return (400, json!({ "error": "speakers must be 1 or more" }));
        }
        let mut held = self.held.lock().unwrap();
        let Some(h) = held.as_mut().filter(|h| h.hash == hash) else {
            return (409, json!({ "error": "the speaker model no longer holds this recording", "stale": true }));
        };
        let t = Instant::now();
        match speakers::recluster(&self.speaker_models(&entry), h, speakers) {
            Ok((turns, split)) => (200, json!({
                "model": entry.id, "speakers": speakers::count(&turns), "turns": turns.len(),
                "cues": speakers::cues(&turns), "hash": hash, "reclustered": true,
                "window_resolution": split, "wall_seconds": round3(t.elapsed().as_secs_f64()),
            })),
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

    /// Whisper over the recording in pieces of up to 30 s, cut at a pause.
    ///
    /// Whisper reads 30 s windows anyway, but whisper.cpp only reports a
    /// window's segments early when DTW word timing is off (its DTW branch
    /// calls new_segment_callback for the first window alone), and DTW timing
    /// is what makes the word boundaries usable for phonetics. Calling it once
    /// per piece keeps DTW and lets each piece's words reach the page as it
    /// finishes, with real progress and a Stop that takes effect between pieces.
    /// Pieces are not prompted with the text before them: in whisper-rs 0.16
    /// both set_initial_prompt (std::bad_alloc on the second piece) and
    /// tokenize (free(): invalid pointer) crash the process. Once the first
    /// piece has detected a language, the rest are held to it.
    ///
    /// The pieces are cut at pauses Silero VAD found (vad.rs), and a piece in
    /// which it heard no speech at all is not given to Whisper: silence is
    /// where Whisper makes text up. A piece whose text repeats itself (see
    /// `looping`) or that has no word Whisper was sure of (`unsure`) is
    /// dropped too, and reported under "warnings".
    fn run(&self, entry: &models::Entry, path: &std::path::Path, bytes: Vec<u8>, filename: &str, language: Option<&str>) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let rate = audio::WHISPER_RATE;
        let duration = samples.len() as f64 / rate as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let ctx = self.context(entry, path)?;
        // whisper wants a two-letter code or nothing. A three-letter code
        // (meant for MMS) would be refused, so fall back to detection.
        let mut lang: Option<String> = language.map(str::trim).filter(|l| l.len() == 2).map(String::from);
        let eot = ctx.token_eot();

        self.stage("recognising words");
        let t = Instant::now();
        let (mut cues, mut text, mut detected) = (Vec::new(), String::new(), String::new());
        let mut warnings = Vec::new();
        for piece in vad::plan(&samples, rate, 30.0) {
            let (at, end) = (piece.range.start, piece.range.end);
            let (offset, piece_len) = (at as f64 / rate as f64, (end - at) as f64 / rate as f64);
            if !piece.speech {
                self.percent.store(((offset + piece_len) / duration.max(1e-3) * 100.0) as u64, Ordering::Relaxed);
                continue;
            }
            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            params.set_language(Some(lang.as_deref().unwrap_or("auto")));
            params.set_n_threads(threads());
            params.set_token_timestamps(true);
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            let pct = self.percent.clone();
            params.set_progress_callback_safe(move |p: i32| {
                let done = offset + piece_len * p.clamp(0, 100) as f64 / 100.0;
                pct.store((done / duration.max(1e-3) * 100.0) as u64, Ordering::Relaxed);
            });
            // Not set_abort_callback_safe: in whisper-rs 0.16 its trampoline reads
            // the boxed closure as the bare closure type, so every call returns
            // garbage. Linux happened to read "continue" on a first run and
            // "abort" later; Windows aborted the encoder at once (error -6). The
            // flag is all the callback needs, so hand whisper.cpp the flag itself,
            // with this piece's deadline (`Abort`), in a local that outlives the
            // full() call below.
            let abort = Abort { cancel: self.cancel.clone(), deadline: Instant::now() + piece_budget(piece_len) };
            unsafe {
                params.set_abort_callback(Some(should_abort));
                params.set_abort_callback_user_data(&abort as *const Abort as *mut std::ffi::c_void);
            }
            // A fresh state per piece: a second full() on one state died with
            // std::bad_alloc before it read any audio. The weights live in ctx,
            // so a state is only its buffers.
            let mut state = ctx.create_state().map_err(|e| e.to_string())?;
            let done = state.full(params, &samples[at..end]);
            if self.cancel.load(Ordering::Relaxed) {
                return Err("stopped".into());
            }
            if let Err(e) = done {
                if Instant::now() < abort.deadline {
                    return Err(format!("whisper failed: {e}"));
                }
                let secs = piece_budget(piece_len).as_secs();
                eprintln!("lana: {} gave up on {offset:.1}-{:.1} s after {secs} s", entry.id, offset + piece_len);
                warnings.push(json!({ "start": round3(offset), "end": round3(offset + piece_len),
                                      "reason": format!("no answer in {secs} s") }));
                continue;
            }
            if lang.is_none() {
                detected = whisper_rs::get_lang_str(state.full_lang_id_from_state()).unwrap_or("").to_string();
                lang = (!detected.is_empty()).then(|| detected.clone());
            }
            let (piece_cues, piece_text) = segments_to_cues(&state, eot, offset, offset + piece_len);
            if let Some(reason) = looping(&piece_text).or_else(|| unsure(&piece_cues)) {
                eprintln!("lana: {} dropped {offset:.1}-{:.1} s ({reason}): {:.80}", entry.id, offset + piece_len, piece_text.trim());
                warnings.push(json!({ "start": round3(offset), "end": round3(offset + piece_len), "reason": reason,
                                      "text": piece_text.trim().chars().take(200).collect::<String>() }));
                continue;
            }
            text.push_str(&piece_text);
            self.done_cues.lock().unwrap().extend(piece_cues.iter().cloned());
            cues.extend(piece_cues);
        }
        eprintln!("lana: {} transcribed {duration:.1} s of audio in {:.1} s", entry.id, t.elapsed().as_secs_f64());
        Ok(json!({
            "text": text.trim(),
            "language": if detected.is_empty() { lang.unwrap_or_default() } else { detected },
            "duration": round3(duration),
            "cues": cues,
            "warnings": warnings,
        }))
    }
}

/// Whether Whisper was guessing over a whole piece: the mean confidence of
/// its words under 0.15. A loop does not always reach the text: whisper.cpp
/// itself notices a looping decode (its entropy test) and retries at higher
/// temperatures, and on the Georgian FLEURS clip whisper-small's last try
/// was "117. external" for all 15 s, at 0.085 and 0.000, after 42 s of
/// retries. Real words, even misheard ones, came out at 0.3-1.0 in testing
/// (whisper-base on the same clip: 0.33-0.66).
fn unsure(cues: &[Value]) -> Option<&'static str> {
    let confs: Vec<f64> = cues.iter()
        .flat_map(|c| c["words"].as_array().into_iter().flatten())
        .filter_map(|w| w["conf"].as_f64())
        .collect();
    let mean = confs.iter().sum::<f64>() / confs.len().max(1) as f64;
    (!confs.is_empty() && mean < 0.15).then_some("no confident words")
}

/// Whether a piece of Whisper's text is a loop rather than speech: Whisper,
/// given a language it knows little or a stretch of noise, can fall into
/// writing one phrase over and over (whisper-small on the Georgian FLEURS clip
/// gave "117. external" for all of it). Two tests, as OpenAI's whisper uses
/// the first: the text compresses far better than speech does (ratio over
/// 2.4), or one n-gram of words repeats back to back.
///
/// The ratio is taken over characters, not UTF-8 bytes: Georgian's three-byte
/// letters share their first two bytes, so an ordinary Georgian sentence
/// already compresses 2.4 to 1 as bytes (1.4 as characters).
pub fn looping(text: &str) -> Option<&'static str> {
    use std::io::Write;
    let text = text.trim();
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 24 {
        let mut ids = std::collections::HashMap::new();
        let bytes: Vec<u8> = if chars.iter().collect::<std::collections::HashSet<_>>().len() < 256 {
            chars.iter().map(|c| { let n = ids.len(); *ids.entry(*c).or_insert(n) as u8 }).collect()
        } else {
            text.as_bytes().to_vec()
        };
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        if z.write_all(&bytes).is_ok() {
            if let Ok(packed) = z.finish() {
                if bytes.len() as f64 / packed.len().max(1) as f64 > 2.4 {
                    return Some("repeated text");
                }
            }
        }
    }
    // An n-gram of n words repeated back to back: eight times for one word
    // ("no, no, no" is speech; eight is not), four times for two or more.
    let words: Vec<String> = text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    for n in 1..=6usize {
        let need = if n == 1 { 8 } else { 4 };
        if words.len() < n * need {
            break;
        }
        for i in 0..=words.len() - n * need {
            if (1..need).all(|k| words[i..i + n] == words[i + k * n..i + k * n + n]) {
                return Some("repeated text");
            }
        }
    }
    None
}

/// One Whisper call's segments as cues, shifted by `offset` seconds and kept
/// inside [offset, limit]. A token that starts with a space starts a word;
/// anything else (word pieces, punctuation) belongs to the last one. A word
/// runs from its first token to the next word's first token.
fn segments_to_cues(state: &whisper_rs::WhisperState, eot: i32, offset: f64, limit: f64) -> (Vec<Value>, String) {
    let mut cues = Vec::new();
    let mut text = String::new();
    for seg in state.as_iter() {
        let a = (offset + seg.start_timestamp() as f64 / 100.0).min(limit);
        let b = (offset + seg.end_timestamp() as f64 / 100.0).min(limit).max(a);
        let seg_text = seg.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default();
        text.push_str(&seg_text);
        // (text, start, sum of token probabilities, tokens)
        let mut words: Vec<(String, f64, f64, u32)> = Vec::new();
        for i in 0..seg.n_tokens() {
            let Some(tok) = seg.get_token(i) else { continue };
            let data = tok.token_data();
            if data.id >= eot {
                continue; // timestamps and control tokens
            }
            let piece = tok.to_str_lossy().map(|s| s.into_owned()).unwrap_or_default();
            let t = offset + if data.t_dtw >= 0 { data.t_dtw as f64 / 100.0 } else { data.t0 as f64 / 100.0 };
            let p = data.p as f64;
            if piece.starts_with(' ') || words.is_empty() {
                words.push((piece.trim_start().to_owned(), t.clamp(a, b), p, 1));
            } else if let Some(last) = words.last_mut() {
                last.0.push_str(&piece);
                last.2 += p;
                last.3 += 1;
            }
        }
        let n = words.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let start = words[k].1;
            let end = if k + 1 < n { words[k + 1].1 } else { b };
            let end = end.max(start + 0.02).min(b.max(start + 0.02));
            // The word's confidence: the mean probability Whisper gave its
            // tokens, punctuation included, as whisper.cpp's own word-level
            // output does.
            let conf = words[k].2 / words[k].3.max(1) as f64;
            out.push(json!({ "text": words[k].0, "start": round3(start), "end": round3(end),
                             "conf": round3(conf), "phones": [], "phone_times": [], "aligned": true }));
        }
        let mut cue = json!({ "start": round3(a), "end": round3(b), "words": out });
        if n == 0 {
            cue["plain"] = json!(seg_text.trim());
        }
        cues.push(cue);
    }
    (cues, text)
}


/// What whisper.cpp's abort callback reads: the job's Stop flag, and when
/// to give up on the piece.
struct Abort {
    cancel: Arc<AtomicBool>,
    deadline: Instant,
}

/// How long one piece may take before Whisper is stopped and the piece left
/// empty. Not for speed: when whisper.cpp finds its own output looping it
/// decodes the piece again at a higher temperature, up to five times, and
/// whisper-small on the 15 s Georgian FLEURS clip once ran for over twenty
/// minutes that way (on a busy machine; 42 s on a quiet one) to return two
/// words. A minute plus five times the piece's length is several times what
/// large-v3 turbo needs on the slowest processor measured (0.7x real time).
fn piece_budget(piece_s: f64) -> std::time::Duration {
    std::time::Duration::from_secs_f64(60.0 + 5.0 * piece_s)
}

/// ggml's abort callback: `user_data` is the piece's `Abort`.
unsafe extern "C" fn should_abort(user_data: *mut std::ffi::c_void) -> bool {
    if user_data.is_null() {
        return false;
    }
    let a = &*(user_data as *const Abort);
    a.cancel.load(Ordering::Relaxed) || Instant::now() >= a.deadline
}

fn preset(id: &str) -> whisper_rs::DtwModelPreset {
    models::find(id).map(|e| e.dtw).unwrap_or(whisper_rs::DtwModelPreset::Base)
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Whether whisper.cpp can run here. Release builds target AVX2, FMA, F16C
/// and BMI2 on x86 (see GGML_NATIVE in .github/workflows/build.yml); on a
/// processor without them it would die with an illegal instruction, taking
/// the app with it. ARM builds have no such floor.
pub fn whisper_supported() -> Result<(), String> {
    #[cfg(target_arch = "x86_64")]
    {
        let ok = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("bmi2");
        if !ok {
            return Err("This processor cannot run Whisper: it lacks AVX2, which Lana's Whisper needs. \
                        The other built-in models (ZIPA, Omnilingual, MMS, Speakers) work.".into());
        }
    }
    Ok(())
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
        .route("/api/local/mms", get(mms_status).delete(mms_delete_all))
        .route("/api/local/mms/accept", post(mms_accept))
        .route("/api/local/mms/languages", get(mms_languages))
        .route("/api/local/mms/languages/{code}", post(mms_download).delete(mms_delete))
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

// ------------------------------------------------------------------ MMS

async fn mms_status(State(l): State<Local>) -> Json<Value> {
    Json(l.models.mms_status())
}

/// The full list, for the language search: [[code, name], ...].
async fn mms_languages() -> Json<Value> {
    Json(json!(mms::languages()))
}

async fn mms_accept(State(l): State<Local>) -> Response {
    match l.models.mms_accept() {
        Ok(()) => Json(l.models.mms_status()).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

async fn mms_download(State(l): State<Local>, UrlPath(code): UrlPath<String>) -> Response {
    match l.models.mms_start(&code) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    }
}

async fn mms_delete(State(l): State<Local>, UrlPath(code): UrlPath<String>) -> Response {
    if l.job.lock().unwrap().as_ref().is_some_and(|j| j.model == format!("mms-{code}")) {
        return (StatusCode::CONFLICT, Json(json!({ "error": "That language is transcribing right now." }))).into_response();
    }
    match l.models.mms_delete(&code) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

async fn mms_delete_all(State(l): State<Local>) -> Response {
    if l.job.lock().unwrap().as_ref().is_some_and(|j| j.model.starts_with("mms-")) {
        return (StatusCode::CONFLICT, Json(json!({ "error": "MMS is transcribing right now." }))).into_response();
    }
    match l.models.mms_delete_all() {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e }))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::looping;

    #[test]
    fn loops_are_caught_and_speech_is_not() {
        assert_eq!(looping(&" 117. external".repeat(12)), Some("repeated text"));
        assert_eq!(looping("ich bin, ich bin, ich bin, ich bin ein Berliner"), Some("repeated text"));
        // Ordinary sentences, Georgian included, whose UTF-8 bytes alone
        // compress 2.4 to 1.
        assert_eq!(looping("ანტარქტიკის უმაღლესი მთები, ვინსონის მასივი, რომლის უმაღლესი მწვერვალი, ვინსონის მთა მდებარეობს 4892 მეტრზე, განლაგებულია სენტინელის ქედის ჩრდილოეთ ნაწილში."), None);
        assert_eq!(looping("Der nördliche Teil oder die Sentinel Range hat die höchsten Berge der Antarktis, das Vinson-Massiv."), None);
        assert_eq!(looping("no, no, no, I said no"), None);
    }
}
