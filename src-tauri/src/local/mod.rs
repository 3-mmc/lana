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
pub mod frames;
pub mod align;
pub mod lid;
pub mod pitch;
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
    /// The last phone pass's frames, kept so search by example can run over
    /// the recording without reading it again.
    kept: Arc<Mutex<Option<align::Kept>>>,
    /// The recording re-timing and search last read, decoded, and the model
    /// they last used: a run of corrections re-times stretch after stretch of
    /// one recording with one model.
    audio: Arc<Mutex<Option<(String, Arc<Vec<f32>>)>>>,
    net: Arc<Mutex<Option<(String, Arc<ctc::Ctc>)>>>,
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
            kept: Default::default(),
            audio: Default::default(),
            net: Default::default(),
        }
    }

    /// The built-in provider, or None until a model has been downloaded.
    pub fn provider(&self) -> Option<Value> {
        let mut models: Vec<Value> = self
            .models
            .installed()
            .iter()
            .filter(|e| e.kind != Kind::Pitch)
            .map(|e| json!({ "id": e.id, "role": e.kind.role(),
                             "label": format!("{} (on this computer{})", e.label, if e.kind.non_commercial() { ", non-commercial" } else { "" }) }))
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
            return self.with_job(model_id, |me| me.run_ctc_at(ctc::Engine::Omnilingual, &dir.join("model.onnx"), &dir.join("tokens.txt"), model_id, req.bytes, &req.filename, None));
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
            Kind::Phones | Kind::Letters => me.run_ctc(&entry, req.bytes, &req.filename, req.language.as_deref()),
            Kind::Transducer => me.run_parakeet(&entry, req.bytes, &req.filename),
            Kind::Tags => me.run_tags(&entry, req.bytes, &req.filename),
            Kind::Lid => me.run_lid(&entry, req.bytes, &req.filename, req.language.as_deref()),
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
        // A recogniser is about to load: free the one re-timing kept.
        *self.net.lock().unwrap() = None;
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

    fn run_ctc(&self, entry: &models::Entry, bytes: Vec<u8>, filename: &str, language: Option<&str>) -> Result<Value, String> {
        let engine = if entry.kind == Kind::Letters { ctc::Engine::Omnilingual } else { ctc::Engine::Zipa };
        let (model, tokens) = (self.models.named(entry, "model.int8.onnx"), self.models.named(entry, "tokens.txt"));
        // Omnilingual is not told the language; its alphabet is the only hold.
        let language = language.filter(|_| engine == ctc::Engine::Omnilingual);
        self.run_ctc_at(engine, &model, &tokens, entry.id, bytes, filename, language)
    }

    /// A CTC model given by its files. MMS runs here too: it has the same
    /// shape as Omnilingual (raw audio in, characters with a word separator out).
    #[allow(clippy::too_many_arguments)]
    fn run_ctc_at(&self, engine: ctc::Engine, model: &std::path::Path, tokens: &std::path::Path, id: &str, bytes: Vec<u8>, filename: &str, language: Option<&str>) -> Result<Value, String> {
        let hash = align::hash(&bytes);
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let duration = samples.len() as f64 / audio::WHISPER_RATE as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let mut rec = ctc::Ctc::load(engine, model, tokens, threads())?;
        rec.lag = ctc::lag_of(id);
        let alphabet = language.and_then(|l| ctc::alphabet(l).map(|a| (l, a)));
        if let Some((_, a)) = alphabet {
            rec.hold_to(a);
        }
        self.stage(if engine == ctc::Engine::Zipa { "reading phones" } else { "recognising words" });
        let t = Instant::now();
        let (pct, cancel, done_cues) = (self.percent.clone(), self.cancel.clone(), self.done_cues.clone());
        let mut pieces = Vec::new();
        let keep = (engine == ctc::Engine::Zipa).then_some(&mut pieces);
        let found = rec.recognise(&samples, audio::WHISPER_RATE, keep, |done, piece| {
            pct.store((done / duration.max(1e-3) * 100.0).clamp(0.0, 100.0) as u64, Ordering::Relaxed);
            let cues = if engine == ctc::Engine::Omnilingual { ctc::word_cues(piece, duration) } else { ctc::cues(piece, duration) };
            done_cues.lock().unwrap().extend(cues);
            !cancel.load(Ordering::Relaxed)
        })?;
        eprintln!("lana: {id} read {} symbols in {duration:.1} s of audio in {:.1} s", found.len(), t.elapsed().as_secs_f64());
        if engine == ctc::Engine::Omnilingual {
            let cues = ctc::word_cues(&found, duration);
            let text = found.iter().map(|c| c.text.as_str()).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
            let mut out = json!({ "text": text, "duration": round3(duration), "cues": cues });
            if let Some((code, _)) = alphabet {
                out["alphabet"] = json!(code);
            }
            return Ok(out);
        }
        *self.kept.lock().unwrap() = Some(align::Kept { hash, model: id.to_owned(), pieces });
        let cues = ctc::cues(&found, duration);
        let text = cues
            .iter()
            .map(|c| c["words"][0]["phones"].as_array().map(|p| p.iter().filter_map(Value::as_str).collect::<String>()).unwrap_or_default())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(json!({ "text": text, "duration": round3(duration), "cues": cues }))
    }

    /// The recording, decoded once per run of corrections.
    fn samples(&self, bytes: Vec<u8>, filename: &str) -> Result<(String, Arc<Vec<f32>>), String> {
        let hash = align::hash(&bytes);
        if let Some((h, s)) = self.audio.lock().unwrap().as_ref().filter(|(h, _)| *h == hash) {
            return Ok((h.clone(), s.clone()));
        }
        let s = Arc::new(audio::decode_16k_mono(bytes, filename)?);
        *self.audio.lock().unwrap() = Some((hash.clone(), s.clone()));
        Ok((hash, s))
    }

    /// A CTC model by its id (zipa-*, omnilingual-*, mms-<code>), loaded
    /// once for a run of corrections.
    fn ctc_model(&self, id: &str) -> Result<Arc<ctc::Ctc>, String> {
        if let Some((_, c)) = self.net.lock().unwrap().as_ref().filter(|(k, _)| k == id) {
            return Ok(c.clone());
        }
        let c = if let Some(code) = id.strip_prefix("mms-") {
            if !self.models.mms_lang_installed(code) {
                return Err(format!("MMS for '{code}' is not downloaded."));
            }
            let dir = self.models.mms_lang_dir(code);
            ctc::Ctc::load(ctc::Engine::Omnilingual, &dir.join("model.onnx"), &dir.join("tokens.txt"), threads())?
        } else {
            let entry = models::find(id).filter(|e| matches!(e.kind, Kind::Phones | Kind::Letters))
                .ok_or_else(|| format!("'{id}' cannot place labels: only ZIPA, Omnilingual and MMS can."))?;
            if !self.models.is_installed(&entry) {
                return Err(format!("{} is not downloaded.", entry.label));
            }
            let engine = if entry.kind == Kind::Letters { ctc::Engine::Omnilingual } else { ctc::Engine::Zipa };
            ctc::Ctc::load(engine, &self.models.named(&entry, "model.int8.onnx"), &self.models.named(&entry, "tokens.txt"), threads())?
        };
        let mut c = c;
        c.lag = ctc::lag_of(id);
        let c = Arc::new(c);
        *self.net.lock().unwrap() = Some((id.to_owned(), c.clone()));
        Ok(c)
    }

    /// Re-time corrected labels: where in [start, end] the model finds each
    /// of `labels`, in order. Phones are placed with a ZIPA model, words with
    /// Omnilingual or MMS (with a word space between them). Each label comes
    /// back with the model's confidence in it, so a correction the model
    /// does not hear stands out; a label with no symbol the model knows is
    /// spread over the gap its neighbours leave.
    pub fn align(&self, bytes: Vec<u8>, filename: &str, model: &str, start: f64, end: f64, labels: &[String]) -> Result<Value, String> {
        let (_, samples) = self.samples(bytes, filename)?;
        let rec = self.ctc_model(model)?;
        let rate = audio::WHISPER_RATE as f64;
        let duration = samples.len() as f64 / rate;
        let (start, end) = (start.max(0.0), end.min(duration));
        if end - start < 0.04 || labels.is_empty() {
            return Err("Select a stretch with labels in it to re-time.".into());
        }
        // Audio either side, so the model hears the stretch in context; the
        // labels are placed inside it only. Omnilingual and MMS normalise
        // over, and attend across, all they are given: with 0.3 s either side
        // the e of "Teil" on the German demo clip read 0.27, against 0.98
        // when the whole clip was read.
        const CONTEXT_S: f64 = 2.0;
        let a = (start - CONTEXT_S).max(0.0);
        let b = (end + CONTEXT_S).min(duration);
        let m = rec.frames(&samples[(a * rate) as usize..(b * rate) as usize])?;
        // A model that runs late hears [start, end] that much later.
        let f0 = ((start + rec.lag - a) / ctc::FRAME_S).ceil() as usize;
        let f1 = ((end + rec.lag - a) / ctc::FRAME_S).floor() as usize;
        let m = m.slice(f0, f1);
        let at = |f: usize| rec.at(a, f0 + f);
        let words = rec.engine == ctc::Engine::Omnilingual;
        let space = rec.symbols().iter().position(|s| s.as_deref() == Some(" "));
        let mut seq = Vec::new();
        let mut owner: Vec<Option<usize>> = Vec::new();
        for (i, label) in labels.iter().enumerate() {
            let ids = align::tokenize(label.trim(), rec.symbols());
            if words && !ids.is_empty() && !seq.is_empty() {
                if let Some(sp) = space {
                    seq.push(sp);
                    owner.push(None);
                }
            }
            owner.extend(std::iter::repeat_n(Some(i), ids.len()));
            seq.extend(ids);
        }
        let placed = align::force(&m, &seq)
            .ok_or("The stretch is too short for these labels, or the model has no symbols for them.")?;
        if std::env::var_os("LANA_DEBUG_ALIGN").is_some() {
            for (k, p) in placed.iter().enumerate() { eprintln!("align {:?} {:?} {:?}", rec.symbols()[seq[k]], owner[k], p); }
        }
        // Each label's first and last frame and its least sure symbol.
        let mut spans: Vec<Option<(usize, usize, f32)>> = vec![None; labels.len()];
        for (k, (first, last, conf)) in placed.iter().enumerate() {
            let Some(i) = owner[k] else { continue };
            spans[i] = Some(match spans[i] {
                None => (*first, *last, *conf),
                Some((f, _, c)) => (f, *last, c.min(*conf)),
            });
        }
        let mut out: Vec<(f64, f64, Option<f32>)> = Vec::new();
        for i in 0..labels.len() {
            let Some((first, last, conf)) = spans[i] else {
                out.push((f64::NAN, f64::NAN, None));
                continue;
            };
            // A label runs to the next one's start; the last, briefly past
            // its last frame.
            let next = spans[i + 1..].iter().flatten().next().map(|s| at(s.0));
            let tail = if words { at(last) + ctc::FRAME_S } else { at(last) + 0.08 };
            let stop = next.unwrap_or_else(|| tail.min(end)).max(at(first) + 0.01);
            out.push((at(first), stop, Some(conf)));
        }
        // Labels the model had no symbol for share the gap where they stand.
        let mut i = 0;
        while i < out.len() {
            if !out[i].0.is_nan() { i += 1; continue; }
            let j = (i..out.len()).find(|&j| !out[j].0.is_nan()).unwrap_or(out.len());
            let from = if i == 0 { start } else { out[i - 1].1 };
            let to = if j < out.len() { out[j].0 } else { end };
            let step = ((to - from) / (j - i) as f64).max(0.0);
            for (n, k) in (i..j).enumerate() {
                out[k] = (from + step * n as f64, from + step * (n + 1) as f64, None);
            }
            i = j;
        }
        let intervals: Vec<Value> = labels.iter().zip(&out).map(|(l, (a, b, c))| json!({
            "text": l, "start": round3(*a), "end": round3(*b), "conf": c.map(|c| round3(c as f64)),
        })).collect();
        Ok(json!({ "model": model, "start": round3(start), "end": round3(end), "intervals": intervals }))
    }

    /// CREPE's f0 and periodicity over the whole recording, every 10 ms.
    pub fn pitch(&self, bytes: Vec<u8>, filename: &str, model: &str) -> Result<Value, String> {
        let entry = models::find(model).filter(|e| e.kind == Kind::Pitch)
            .ok_or_else(|| format!("'{model}' is not a pitch model."))?;
        if !self.models.is_installed(&entry) {
            return Err(format!("{} is not downloaded. Add it in Connection → Recognisers.", entry.label));
        }
        let (_, samples) = self.samples(bytes, filename)?;
        let t = Instant::now();
        let crepe = pitch::Crepe::load(&self.models.path(&entry), threads())?;
        let cancel = self.cancel.clone();
        let track = crepe.track(&samples, |_| !cancel.load(Ordering::Relaxed))?;
        eprintln!("lana: {model} tracked {:.1} s of audio in {:.1} s", samples.len() as f64 / audio::WHISPER_RATE as f64, t.elapsed().as_secs_f64());
        let f0: Vec<f64> = track.f0.iter().map(|f| (*f as f64 * 10.0).round() / 10.0).collect();
        let conf: Vec<f64> = track.conf.iter().map(|c| round3(*c as f64)).collect();
        Ok(json!({ "model": model, "hop": pitch::HOP_S, "f0": f0, "conf": conf }))
    }

    /// Search by example: the places in the recording most like the phones of
    /// [start, end], or like `phones` as typed. Reads the whole recording with
    /// ZIPA the first time, unless a phone pass just did.
    pub fn search(&self, bytes: Vec<u8>, filename: &str, model: &str, span: Option<(f64, f64)>, phones: Option<&str>, max: usize) -> Result<Value, String> {
        let (hash, samples) = self.samples(bytes, filename)?;
        let rec = self.ctc_model(model)?;
        if rec.engine != ctc::Engine::Zipa {
            return Err("Search by example reads phones: choose a ZIPA model.".into());
        }
        let mut kept = self.kept.lock().unwrap();
        if !kept.as_ref().is_some_and(|k| k.hash == hash && k.model == model) {
            let mut pieces = Vec::new();
            let cancel = self.cancel.clone();
            rec.recognise(&samples, audio::WHISPER_RATE, Some(&mut pieces), |_, _| !cancel.load(Ordering::Relaxed))?;
            *kept = Some(align::Kept { hash, model: model.to_owned(), pieces });
        }
        let k = kept.as_ref().unwrap();
        let query: Vec<String> = match (phones, span) {
            (Some(p), _) => align::tokenize(p, rec.symbols()).iter().filter_map(|&i| rec.symbols()[i].clone()).collect(),
            (None, Some((a, b))) => k.pieces.iter().flat_map(|p| {
                let from = ((a - p.offset) / ctc::FRAME_S).max(0.0) as usize;
                let to = ((b - p.offset) / ctc::FRAME_S).max(0.0) as usize;
                rec.decode(&p.mat.slice(from, to), p.offset + from as f64 * ctc::FRAME_S)
            }).map(|s| s.text).collect(),
            (None, None) => return Err("Select a stretch, or type the phones to look for.".into()),
        };
        let ids: Vec<usize> = query.iter().flat_map(|q| align::tokenize(q, rec.symbols())).collect();
        if ids.is_empty() {
            return Err("No phones were heard in the selection.".into());
        }
        let hits = align::spot(&k.pieces, &ids, max, 0.05);
        let found: Vec<Value> = hits.iter().map(|h| {
            let own = span.is_some_and(|(a, b)| h.start < b && h.end > a);
            json!({ "start": round3(h.start), "end": round3(h.end), "score": round3(h.score as f64), "selection": own })
        }).collect();
        Ok(json!({ "model": model, "query": query, "hits": found }))
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

    /// Language ID over the recording, a window at a time (lid.rs). For this
    /// model `language` is the languages the linguist expects, comma
    /// separated; the classifier then chooses among those alone.
    fn run_lid(&self, entry: &models::Entry, bytes: Vec<u8>, filename: &str, languages: Option<&str>) -> Result<Value, String> {
        let samples = audio::decode_16k_mono(bytes, filename)?;
        let duration = samples.len() as f64 / lid::RATE as f64;
        if let Some(j) = self.job.lock().unwrap().as_mut() {
            j.total = duration;
        }
        self.stage("loading model");
        let model = lid::Lid::load(&self.models.named(entry, "model.int8.onnx"), &self.models.named(entry, "config.json"), threads())?;
        let codes: Vec<String> = languages.unwrap_or("").split(',').map(|c| c.trim().to_owned()).filter(|c| !c.is_empty()).collect();
        let (cand, unknown) = model.candidates(&codes);
        self.stage("telling languages apart");
        let t = Instant::now();
        let speech = vad::speech(&samples, lid::RATE as u32);
        let (pct, cancel) = (self.percent.clone(), self.cancel.clone());
        let found = model.stretches(&samples, speech.as_deref(), &cand, |done| {
            pct.store((done / duration.max(1e-3) * 100.0).clamp(0.0, 100.0) as u64, Ordering::Relaxed);
            !cancel.load(Ordering::Relaxed)
        })?;
        eprintln!("lana: {} found {} language stretches in {duration:.1} s of audio in {:.1} s ({} candidates)",
                  entry.id, found.len(), t.elapsed().as_secs_f64(), cand.len());
        let cues = lid::cues(&found);
        self.done_cues.lock().unwrap().extend(cues.iter().cloned());
        let mut langs: Vec<&str> = found.iter().map(|s| s.lang.as_str()).collect();
        langs.dedup();
        let warnings: Vec<Value> = unknown.iter()
            .map(|c| json!({ "start": 0.0, "end": round3(duration), "reason": format!("{} does not know the language '{c}'", entry.label) }))
            .collect();
        Ok(json!({ "text": langs.join(", "), "duration": round3(duration), "cues": cues, "warnings": warnings }))
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
    extract::{Multipart, Path as UrlPath, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};

/// Re-timing and search take the recording itself, as /api/transcribe does,
/// and allow the same size (broker.rs; local/ cannot name it, see bench/).
const MAX_UPLOAD: usize = 512 * 1024 * 1024;

/// The settings screen's model manager. Desktop only, like the rest of settings.
pub fn router(local: Local) -> Router {
    Router::new()
        .route("/api/local/models", get(list_models))
        .route("/api/local/models/{id}", post(download_model).delete(delete_model))
        .route("/api/local/mms", get(mms_status).delete(mms_delete_all))
        .route("/api/local/mms/accept", post(mms_accept))
        .route("/api/local/mms/languages", get(mms_languages))
        .route("/api/local/mms/languages/{code}", post(mms_download).delete(mms_delete))
        .route("/api/local/align", post(align_labels))
        .route("/api/local/search", post(search_phones))
        .route("/api/local/pitch", post(pitch_track))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_UPLOAD))
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

// ------------------------------------------------- re-timing and search

/// A multipart form as its audio, the audio's name, and its text fields.
async fn form_of(mut form: Multipart) -> Result<(Vec<u8>, String, std::collections::HashMap<String, String>), String> {
    let (mut bytes, mut name, mut fields) = (None, "audio.wav".to_owned(), std::collections::HashMap::new());
    while let Some(part) = form.next_field().await.map_err(|e| format!("bad upload: {e}"))? {
        let Some(key) = part.name().map(str::to_owned) else { continue };
        if key == "file" {
            if let Some(f) = part.file_name() {
                name = f.to_owned();
            }
            bytes = Some(part.bytes().await.map_err(|e| format!("upload failed: {e}"))?.to_vec());
        } else if let Ok(text) = part.text().await {
            fields.insert(key, text);
        }
    }
    Ok((bytes.filter(|b| !b.is_empty()).ok_or("file is required")?, name, fields))
}

fn num(fields: &std::collections::HashMap<String, String>, k: &str) -> Option<f64> {
    fields.get(k).and_then(|v| v.trim().parse().ok())
}

/// Fields: file, model, start, end, labels (a JSON array of strings).
async fn align_labels(State(l): State<Local>, form: Multipart) -> Response {
    let (bytes, name, f) = match form_of(form).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    };
    let labels: Vec<String> = f.get("labels").and_then(|v| serde_json::from_str(v).ok()).unwrap_or_default();
    let (model, start, end) = (f.get("model").cloned().unwrap_or_default(), num(&f, "start").unwrap_or(0.0), num(&f, "end").unwrap_or(0.0));
    let r = tokio::task::spawn_blocking(move || l.align(bytes, &name, &model, start, end, &labels)).await;
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("re-timing crashed: {e}") }))).into_response(),
    }
}

/// Fields: file, model (a ZIPA id), and either start and end, or phones; max.
async fn search_phones(State(l): State<Local>, form: Multipart) -> Response {
    let (bytes, name, f) = match form_of(form).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    };
    let model = f.get("model").cloned().unwrap_or_default();
    let span = num(&f, "start").zip(num(&f, "end"));
    let phones = f.get("phones").map(|p| p.trim().to_owned()).filter(|p| !p.is_empty());
    let max = num(&f, "max").map_or(20, |m| m.clamp(1.0, 200.0) as usize);
    let r = tokio::task::spawn_blocking(move || l.search(bytes, &name, &model, span, phones.as_deref(), max)).await;
    match r {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("search crashed: {e}") }))).into_response(),
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

/// Fields: file, model (a CREPE id).
async fn pitch_track(State(l): State<Local>, form: Multipart) -> Response {
    let (bytes, name, f) = match form_of(form).await {
        Ok(v) => v,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e }))).into_response(),
    };
    let model = f.get("model").cloned().unwrap_or_default();
    match tokio::task::spawn_blocking(move || l.pitch(bytes, &name, &model)).await {
        Ok(Ok(v)) => Json(v).into_response(),
        Ok(Err(e)) => (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "error": e }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": format!("the pitch model crashed: {e}") }))).into_response(),
    }
}
