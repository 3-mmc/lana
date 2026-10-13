// SPDX-License-Identifier: GPL-3.0-or-later
// The recognition broker: a port of serve.py's /api/* surface.
//
// The page cannot tell this from serve.py, and that is the test: the same
// requests against both must give the same JSON (scripts/parity-test.py).
// Where this differs from serve.py on purpose, the comment says so.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant, SystemTime},
};

use axum::{
    extract::{DefaultBodyLimit, Multipart, Query, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Map, Value};

const MAX_UPLOAD: usize = 512 * 1024 * 1024;
const ASR_TIMEOUT: Duration = Duration::from_secs(3600); // a film is an hour of GPU work
const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
/// Service name for API keys in the OS keychain; the account is the provider id.
pub const KEYCHAIN_SERVICE: &str = "lana";

/// Fields our own services understand. Anything not listed is dropped on the way
/// through, which looks exactly like a model ignoring an argument, so add the
/// name here (and in serve.py) when the page grows one.
const EXTRAS: &[&str] = &[
    "asr_model", "translate_to", "speakers", "threshold", "min_on", "min_off", "embedding",
];

#[derive(Clone)]
pub struct Broker {
    file: PathBuf,
    cache: Arc<Mutex<(Option<SystemTime>, Vec<Value>)>>,
    http: reqwest::Client,
}

pub fn router(file: PathBuf) -> Router {
    let broker = Broker {
        file,
        cache: Arc::new(Mutex::new((None, Vec::new()))),
        http: reqwest::Client::builder()
            .user_agent(concat!("lana-desktop/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("HTTP client"),
    };
    Router::new()
        .route("/api/health", get(health))
        .route("/api/providers", get(providers))
        .route("/api/progress", get(progress))
        .route("/api/partial", get(partial))
        .route("/api/cancel", post(cancel))
        .route("/api/recluster", post(recluster))
        .route("/api/transcribe", post(transcribe))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD))
        .with_state(broker)
}

fn reply(code: u16, v: Value) -> Response {
    (StatusCode::from_u16(code).unwrap_or(StatusCode::BAD_GATEWAY), Json(v)).into_response()
}

// ------------------------------------------------------------- providers

impl Broker {
    /// providers.json, re-read whenever it changes on disk: someone adding a key
    /// and wondering why nothing noticed is the common case, and a stat() per
    /// request is cheaper than explaining a restart.
    fn load(&self) -> Vec<Value> {
        let mtime = std::fs::metadata(&self.file).and_then(|m| m.modified()).ok();
        let mut cache = self.cache.lock().unwrap();
        if mtime.is_none() {
            *cache = (None, Vec::new());
            return Vec::new();
        }
        if cache.0 == mtime {
            return cache.1.clone();
        }
        let value = std::fs::read_to_string(&self.file)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str::<Value>(&s).map_err(|e| e.to_string()))
            .map(|doc| {
                doc.get("providers")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter(|p| !str_of(p, "id").is_empty()).cloned().collect())
                    .unwrap_or_default()
            })
            .unwrap_or_else(|e| {
                eprintln!("providers.json is not usable: {e}");
                Vec::new()
            });
        *cache = (mtime, value.clone());
        value
    }

    fn find(&self, pid: &str) -> Option<Value> {
        self.load().into_iter().find(|p| str_of(p, "id") == pid)
    }

    fn pick(&self, pid: Option<&str>) -> Result<Value, Response> {
        let pid = pid.unwrap_or("");
        self.find(pid)
            .ok_or_else(|| reply(404, json!({ "error": format!("no provider '{pid}' is configured") })))
    }
}

fn str_of<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

/// The provider's API key, or None. An explicit key in the file beats the
/// keychain, which beats the environment. (serve.py has no keychain step.)
fn provider_key(p: &Value) -> Option<String> {
    let nonempty = |s: String| (!s.is_empty()).then_some(s);
    if let Some(k) = p.get("api_key").and_then(Value::as_str) {
        if !k.is_empty() {
            return Some(k.to_owned());
        }
    }
    if p.get("keychain").and_then(Value::as_bool).unwrap_or(false) {
        if let Some(k) = keychain_get(str_of(p, "id")) {
            return Some(k);
        }
    }
    let env = str_of(p, "api_key_env");
    if env.is_empty() {
        None
    } else {
        std::env::var(env).ok().and_then(nonempty)
    }
}

pub fn keychain_get(account: &str) -> Option<String> {
    keyring::Entry::new(KEYCHAIN_SERVICE, account)
        .and_then(|e| e.get_password())
        .ok()
        .filter(|k| !k.is_empty())
}

/// What the page may know. No key, ever, not even its length.
fn public_provider(p: &Value) -> Value {
    let keychain = p.get("keychain").and_then(Value::as_bool).unwrap_or(false);
    let needs_key = !str_of(p, "api_key").is_empty() || !str_of(p, "api_key_env").is_empty() || keychain;
    let has_key = provider_key(p).is_some();
    let note = if needs_key && !has_key {
        if keychain {
            "no key: add one in Settings".to_owned()
        } else {
            let env = str_of(p, "api_key_env");
            format!("no key: set {} and reload", if env.is_empty() { "api_key" } else { env })
        }
    } else {
        String::new()
    };
    let id = str_of(p, "id");
    let label = str_of(p, "label");
    json!({
        "id": id,
        "label": if label.is_empty() { id } else { label },
        "kind": p.get("kind").and_then(Value::as_str).unwrap_or("openai"),
        "base": str_of(p, "base"),
        "models": p.get("models").cloned().unwrap_or_else(|| json!([])),
        "control": !str_of(p, "control").is_empty(),
        "needs_key": needs_key,
        "has_key": has_key,
        "ready": has_key || !needs_key,
        "note": note,
    })
}

/// The control base for a provider, or for one model of it: a gateway that
/// swaps models puts each service behind its own path, so a model may override
/// the provider's `control`.
fn control_url(p: &Value, path: &str, model: Option<&str>) -> Option<String> {
    let mut base = str_of(p, "control");
    if let Some(model) = model {
        if let Some(m) = p.get("models").and_then(Value::as_array).and_then(|ms| {
            ms.iter().find(|m| str_of(m, "id") == model && !str_of(m, "control").is_empty())
        }) {
            base = str_of(m, "control");
        }
    }
    if base.is_empty() {
        return None;
    }
    Some(format!(
        "{}/{}{}",
        str_of(p, "base").trim_end_matches('/'),
        base.trim_matches('/'),
        path
    ))
}

// -------------------------------------------------------------- handlers

async fn health(State(b): State<Broker>) -> Json<Value> {
    Json(json!({ "status": "ok", "service": "lana-desktop", "providers": b.load().len() }))
}

async fn providers(State(b): State<Broker>) -> Json<Value> {
    let ps: Vec<Value> = b.load().iter().map(public_provider).collect();
    let configured = (!ps.is_empty()).then(|| b.file.display().to_string());
    Json(json!({ "providers": ps, "configured": configured }))
}

type Q = Query<std::collections::HashMap<String, String>>;

async fn progress(State(b): State<Broker>, Query(q): Q) -> Response {
    poll(&b, &q, "/progress".into()).await
}

async fn partial(State(b): State<Broker>, Query(q): Q) -> Response {
    let since = q.get("since").map(String::as_str).unwrap_or("0");
    poll(&b, &q, format!("/partial?since={}", urlencoding(since))).await
}

/// A poll that fails is answered as "nothing running", never as an error: the
/// page polls on a timer and a provider without progress is normal.
async fn poll(b: &Broker, q: &std::collections::HashMap<String, String>, tail: String) -> Response {
    let p = match b.pick(q.get("provider").map(String::as_str)) {
        Ok(p) => p,
        Err(r) => return r,
    };
    if str_of(&p, "control").is_empty() {
        return reply(200, json!({ "active": false, "unsupported": true }));
    }
    match proxy_control(b, &p, &tail, false, None, None).await {
        (200, body) => reply(200, body),
        _ => reply(200, json!({ "active": false, "unsupported": true })),
    }
}

async fn cancel(State(b): State<Broker>, Query(q): Q) -> Response {
    let p = match b.pick(q.get("provider").map(String::as_str)) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let (code, body) = proxy_control(&b, &p, "/cancel", true, None, None).await;
    reply(code, body)
}

/// A re-cluster carries no audio: the diarizer still holds the recording from
/// the pass that produced this hash.
async fn recluster(State(b): State<Broker>, body: axum::body::Bytes) -> Response {
    if body.is_empty() || body.len() > 64 * 1024 {
        return reply(400, json!({ "error": "expected a small JSON body" }));
    }
    let want: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return reply(400, json!({ "error": format!("bad JSON: {e}") })),
    };
    let p = match b.pick(want.get("provider").and_then(Value::as_str)) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let send = json!({ "hash": want.get("hash"), "speakers": want.get("speakers") });
    let model = want.get("model").and_then(Value::as_str);
    let (code, body) = proxy_control(&b, &p, "/recluster", true, model, Some(send)).await;
    reply(code, body)
}

async fn transcribe(State(b): State<Broker>, mut form: Multipart) -> Response {
    let mut fields = Map::new();
    let (mut blob, mut filename, mut filetype) = (None, "audio.wav".to_owned(), None);
    loop {
        let part = match form.next_field().await {
            Ok(Some(part)) => part,
            Ok(None) => break,
            Err(e) => return reply(400, json!({ "error": format!("bad upload: {e}") })),
        };
        let Some(name) = part.name().map(str::to_owned) else { continue };
        if name == "file" {
            if let Some(f) = part.file_name() {
                filename = f.to_owned();
            }
            filetype = part.content_type().map(str::to_owned);
            match part.bytes().await {
                Ok(bytes) => blob = Some(bytes),
                Err(e) => return reply(413, json!({ "error": format!("upload failed: {e}") })),
            }
        } else if let Ok(text) = part.text().await {
            fields.insert(name, Value::String(text));
        }
    }
    let Some(blob) = blob.filter(|b| !b.is_empty()) else {
        return reply(400, json!({ "error": "file is required" }));
    };
    let filetype = filetype
        .or_else(|| mime_guess::from_path(&filename).first().map(|m| m.to_string()))
        .unwrap_or_else(|| "application/octet-stream".into());

    let p = match b.pick(fields.get("provider").and_then(Value::as_str)) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let public = public_provider(&p);
    if public["ready"] != json!(true) {
        return reply(400, json!({ "error": public["note"] }));
    }
    let started = Instant::now();
    let (code, mut payload) = forward_transcription(&b, &p, &fields, filename, filetype, blob).await;
    if code == 200 {
        if let Some(obj) = payload.as_object_mut() {
            obj.entry("provider").or_insert_with(|| json!(str_of(&p, "id")));
            obj.entry("model").or_insert_with(|| fields.get("model").cloned().unwrap_or(Value::Null));
            let secs = (started.elapsed().as_secs_f64() * 100.0).round() / 100.0;
            obj.insert("wall_seconds".into(), json!(secs));
        }
    }
    reply(code, payload)
}

// ------------------------------------------------------------ forwarding

async fn forward_transcription(
    b: &Broker,
    p: &Value,
    fields: &Map<String, Value>,
    filename: String,
    filetype: String,
    blob: axum::body::Bytes,
) -> (u16, Value) {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("openai");
    if kind != "openai" {
        return (400, json!({ "error": format!("provider kind '{kind}' is not implemented") }));
    }
    let field = |k: &str| fields.get(k).and_then(Value::as_str).filter(|s| !s.is_empty());
    let mut send: Vec<(String, String)> = Vec::new();
    if let Some(m) = field("model") {
        send.push(("model".into(), m.into()));
    }
    send.push(("response_format".into(), field("response_format").unwrap_or("verbose_json").into()));
    if let Some(l) = field("language") {
        send.push(("language".into(), l.into()));
    }
    // Extras are opt-out per provider: OpenAI rejects unknown fields on some routes.
    if p.get("pass_extras").and_then(Value::as_bool).unwrap_or(true) {
        for extra in EXTRAS {
            if let Some(v) = field(extra) {
                send.push(((*extra).into(), v.into()));
            }
        }
    }
    if let Some(fixed) = p.get("fields").and_then(Value::as_object) {
        for (k, v) in fixed {
            let v = v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string());
            send.retain(|(name, _)| name != k);
            send.push((k.clone(), v));
        }
    }

    let mut form = reqwest::multipart::Form::new();
    for (k, v) in send {
        form = form.text(k, v);
    }
    let file = match reqwest::multipart::Part::bytes(blob.to_vec()).file_name(filename).mime_str(&filetype) {
        Ok(f) => f,
        Err(e) => return (400, json!({ "error": format!("bad file type '{filetype}': {e}") })),
    };
    form = form.part("file", file);

    let url = format!("{}/v1/audio/transcriptions", str_of(p, "base").trim_end_matches('/'));
    let mut req = b.http.post(&url).multipart(form).timeout(ASR_TIMEOUT);
    req = with_auth(req, p);
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return (502, json!({ "error": format!("cannot reach {} at {}: {e}", str_of(p, "id"), str_of(p, "base")) }))
        }
    };
    let status = resp.status().as_u16();
    let raw = resp.bytes().await.unwrap_or_default();
    if status >= 400 {
        return (status, json!({ "error": error_detail(&raw), "provider": str_of(p, "id") }));
    }
    match serde_json::from_slice::<Value>(&raw) {
        Ok(v) => (200, normalise(v)),
        Err(_) => (200, json!({ "text": String::from_utf8_lossy(&raw), "cues": [] })),
    }
}

fn with_auth(mut req: reqwest::RequestBuilder, p: &Value) -> reqwest::RequestBuilder {
    if let Some(key) = provider_key(p) {
        req = req.bearer_auth(key);
    }
    if let Some(h) = p.get("headers").and_then(Value::as_object) {
        for (k, v) in h {
            if let Some(v) = v.as_str() {
                req = req.header(k, v);
            }
        }
    }
    req
}

/// The upstream's error as one readable string, as serve.py extracts it.
fn error_detail(raw: &[u8]) -> Value {
    let text: String = String::from_utf8_lossy(raw).chars().take(600).collect();
    match serde_json::from_str::<Value>(&text).ok().and_then(|v| v.get("error").cloned()) {
        Some(Value::Object(o)) => o
            .get("message")
            .cloned()
            .unwrap_or_else(|| json!(Value::Object(o).to_string().chars().take(600).collect::<String>())),
        Some(e) => e,
        None => json!(text),
    }
}

async fn proxy_control(
    b: &Broker,
    p: &Value,
    path: &str,
    post: bool,
    model: Option<&str>,
    body: Option<Value>,
) -> (u16, Value) {
    let Some(url) = control_url(p, path, model) else {
        return (404, json!({ "error": "this provider declares no control endpoint" }));
    };
    let mut req = if post { b.http.post(&url) } else { b.http.get(&url) }.timeout(CONTROL_TIMEOUT);
    if let Some(body) = body {
        req = req.json(&body);
    } else if post {
        req = req.header("Content-Length", "0");
    }
    req = with_auth(req, p);
    let result = async {
        let r = req.send().await?.error_for_status()?;
        let bytes = r.bytes().await?;
        Ok::<_, reqwest::Error>(bytes)
    }
    .await;
    match result {
        Ok(bytes) if bytes.is_empty() => (200, json!({})),
        Ok(bytes) => match serde_json::from_slice(&bytes) {
            Ok(v) => (200, v),
            Err(e) => (502, json!({ "error": e.to_string() })),
        },
        Err(e) => (502, json!({ "error": e.to_string() })),
    }
}

fn urlencoding(s: &str) -> String {
    s.bytes()
        .map(|c| match c {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => (c as char).to_string(),
            _ => format!("%{c:02X}"),
        })
        .collect()
}

// ------------------------------------------------------------- normalise

/// Coerce a transcription response into the cue contract. Our own services
/// already speak it; OpenAI-shaped ones return `segments`, plus a flat `words`
/// list (only if word timestamps were asked for) that has to be put back into
/// its segments. A response with neither still transcribes but draws no tier.
pub fn normalise(payload: Value) -> Value {
    let Value::Object(obj) = payload else {
        return json!({ "text": "", "cues": [] });
    };
    if obj.contains_key("cues") {
        return Value::Object(obj);
    }
    let num = |v: Option<&Value>| v.and_then(|x| x.as_f64().or_else(|| x.as_str()?.parse().ok()));
    let segs: Vec<&Value> = obj.get("segments").and_then(Value::as_array).map(|a| a.iter().collect()).unwrap_or_default();
    let words: Vec<(f64, f64, &Value)> = obj
        .get("words")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|w| Some((num(w.get("start"))?, num(w.get("end"))?, w)))
                .collect()
        })
        .unwrap_or_default();
    let word_obj = |(s, e, w): &(f64, f64, &Value)| {
        let text = [str_of(w, "word"), str_of(w, "text")].into_iter().find(|t| !t.is_empty()).unwrap_or("");
        json!({ "text": text, "start": s, "end": e, "phones": [], "phone_times": [], "aligned": true })
    };

    let mut cues = Vec::new();
    if !segs.is_empty() {
        let bounds: Vec<(f64, f64)> = segs
            .iter()
            .map(|s| (num(s.get("start")).unwrap_or(0.0), num(s.get("end")).unwrap_or(0.0)))
            .collect();
        let mut buckets: Vec<Vec<&(f64, f64, &Value)>> = vec![Vec::new(); segs.len()];
        for w in &words {
            let mid = (w.0 + w.1) / 2.0;
            let idx = bounds.iter().position(|&(a, b)| a <= mid && mid <= b).unwrap_or_else(|| {
                // between segments: give it the nearest one (first wins a tie, as min() does)
                let dist = |i: usize| (mid - bounds[i].0).abs().min((mid - bounds[i].1).abs());
                (0..bounds.len()).fold(0, |best, i| if dist(i) < dist(best) { i } else { best })
            });
            buckets[idx].push(w);
        }
        for ((s, (a, b)), ws) in segs.iter().zip(bounds).zip(buckets) {
            let mut cue = json!({ "start": a, "end": b, "words": ws.iter().map(|w| word_obj(w)).collect::<Vec<_>>() });
            if ws.is_empty() {
                cue["plain"] = json!(str_of(s, "text").trim());
            }
            cues.push(cue);
        }
    } else if !words.is_empty() {
        cues.push(json!({
            "start": words[0].0,
            "end": words[words.len() - 1].1,
            "words": words.iter().map(word_obj).collect::<Vec<_>>(),
        }));
    }
    json!({
        "text": obj.get("text").cloned().unwrap_or(json!("")),
        "language": obj.get("language").cloned().unwrap_or(Value::Null),
        "duration": obj.get("duration").cloned().unwrap_or(Value::Null),
        "cues": cues,
    })
}
