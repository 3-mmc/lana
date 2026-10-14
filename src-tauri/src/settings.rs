// SPDX-License-Identifier: GPL-3.0-or-later
// Desktop-only settings: edit providers.json and API keys from the page.
//
// None of this exists in serve.py, and the page shows the settings screen only
// when GET /api/capabilities says `desktop: true`. Keys go in one direction:
// the page can set or delete one, never read it back.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use serde_json::{json, Map, Value};

use crate::broker::KEYCHAIN_SERVICE;

#[derive(Clone)]
struct Settings {
    file: PathBuf,
    /// The recording the app was launched with, if any. The only file outside
    /// the app's own resources that this server will hand to the page.
    launch: Option<PathBuf>,
    write: Arc<Mutex<()>>,
    http: reqwest::Client,
}

pub fn router(file: PathBuf, launch: Option<PathBuf>) -> Router {
    let s = Settings {
        file,
        launch,
        write: Arc::new(Mutex::new(())),
        http: reqwest::Client::builder()
            .user_agent(concat!("lana-desktop/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("HTTP client"),
    };
    Router::new()
        .route("/api/capabilities", get(capabilities))
        .route("/api/launch-file", get(launch_file))
        .route("/api/log", post(log))
        .route("/api/settings/providers", get(list).post(upsert))
        .route("/api/settings/providers/{id}", delete(remove))
        .route("/api/settings/keys/{id}", put(set_key).delete(delete_key))
        .route("/api/settings/test", post(test))
        .with_state(s)
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

// ------------------------------------------------------------- keychain

/// What this platform calls its password store, in words a user would recognise.
fn keychain_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "the macOS Keychain"
    } else if cfg!(target_os = "windows") {
        "Windows Credential Manager"
    } else {
        "the system keyring"
    }
}

/// Whether a key can be stored at all. Linux without a Secret Service daemon
/// (WSL, minimal window managers) has nowhere to put one; reading an entry that
/// does not exist answers NoEntry only when there is a store to ask.
fn keychain_available() -> bool {
    match keyring::Entry::new(KEYCHAIN_SERVICE, "__lana_probe__").and_then(|e| e.get_password()) {
        Ok(_) | Err(keyring::Error::NoEntry) => true,
        Err(_) => false,
    }
}

async fn capabilities(State(s): State<Settings>) -> Json<Value> {
    let available = tokio::task::spawn_blocking(keychain_available).await.unwrap_or(false);
    Json(json!({
        "desktop": true,
        // Annotations and settings are kept by the app (/api/store), because
        // browser storage does not survive the port changing between launches.
        "store": true,
        "version": env!("CARGO_PKG_VERSION"),
        "keychain": available.then_some(keychain_name()),
        "providers_file": s.file.display().to_string(),
        "launch_file": s.launch.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy()),
    }))
}

/// The page's errors, printed where a developer running the app can see them.
/// A webview's console is otherwise out of reach in a release build.
async fn log(body: axum::body::Bytes) -> StatusCode {
    let text = String::from_utf8_lossy(&body[..body.len().min(4096)]);
    eprintln!("page: {text}");
    StatusCode::NO_CONTENT
}

async fn launch_file(State(s): State<Settings>) -> Response {
    let Some(path) = s.launch.clone() else {
        return err(StatusCode::NOT_FOUND, "the app was not opened with a file");
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => {
            let mime = mime_guess::from_path(&path).first_or_octet_stream().to_string();
            ([(axum::http::header::CONTENT_TYPE, mime)], bytes).into_response()
        }
        Err(e) => err(StatusCode::NOT_FOUND, format!("cannot read {}: {e}", path.display())),
    }
}

// ------------------------------------------------------------- the file

impl Settings {
    /// The whole document, so `_comment` and anything else a hand-editor put
    /// there survives a save from the page.
    fn read(&self) -> Result<Map<String, Value>, String> {
        match std::fs::read_to_string(&self.file) {
            Ok(text) => match serde_json::from_str::<Value>(&text) {
                Ok(Value::Object(doc)) => Ok(doc),
                Ok(_) => Err("providers.json is not a JSON object".into()),
                Err(e) => Err(format!("providers.json cannot be read: {e}")),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Map::new()),
            Err(e) => Err(format!("providers.json cannot be opened: {e}")),
        }
    }

    /// Write to a sibling and rename, so a crash mid-save cannot leave half a file.
    fn write(&self, doc: &Map<String, Value>) -> Result<(), String> {
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let tmp = self.file.with_extension("json.saving");
        let text = serde_json::to_string_pretty(&Value::Object(doc.clone())).unwrap() + "\n";
        std::fs::write(&tmp, text).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, &self.file).map_err(|e| format!("cannot replace {}: {e}", self.file.display()))
    }

    /// Read, change, write, under one lock.
    fn edit<T>(&self, f: impl FnOnce(&mut Vec<Value>) -> Result<T, Response>) -> Result<T, Response> {
        let _guard = self.write.lock().unwrap();
        let mut doc = self.read().map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let mut providers = doc.get("providers").and_then(Value::as_array).cloned().unwrap_or_default();
        let out = f(&mut providers)?;
        doc.insert("providers".into(), Value::Array(providers));
        self.write(&doc).map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        Ok(out)
    }
}

fn id_of(p: &Value) -> &str {
    p.get("id").and_then(Value::as_str).unwrap_or("")
}

/// The editable view: everything but key material.
fn redact(p: &Value) -> Value {
    let mut p = p.clone();
    if let Some(o) = p.as_object_mut() {
        let in_file = o.remove("api_key").is_some_and(|k| k.as_str().is_some_and(|k| !k.is_empty()));
        o.insert("key_in_file".into(), json!(in_file));
    }
    p
}

async fn list(State(s): State<Settings>) -> Response {
    match s.read() {
        Ok(doc) => {
            let ps: Vec<Value> = doc
                .get("providers")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(redact).collect())
                .unwrap_or_default();
            Json(json!({ "providers": ps })).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// Add a provider, or replace the one with the same id. A key already stored
/// for it is kept: the page never has it to send back.
async fn upsert(State(s): State<Settings>, Json(mut p): Json<Value>) -> Response {
    let Some(obj) = p.as_object_mut() else {
        return err(StatusCode::BAD_REQUEST, "expected a provider object");
    };
    obj.remove("key_in_file");
    obj.remove("api_key");
    let id = obj.get("id").and_then(Value::as_str).unwrap_or("").to_owned();
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return err(StatusCode::BAD_REQUEST, "a provider needs an id of letters, digits, - and _");
    }
    let base = obj.get("base").and_then(Value::as_str).unwrap_or("");
    if reqwest::Url::parse(base).map_or(true, |u| !matches!(u.scheme(), "http" | "https")) {
        return err(StatusCode::BAD_REQUEST, format!("'{base}' is not a web address (it should start with http:// or https://)"));
    }
    let result = s.edit(|ps| {
        if let Some(old) = ps.iter_mut().find(|q| id_of(q) == id) {
            for keep in ["api_key", "keychain"] {
                if let Some(v) = old.get(keep).cloned() {
                    obj.entry(keep).or_insert(v);
                }
            }
            *old = Value::Object(obj.clone());
        } else {
            ps.push(Value::Object(obj.clone()));
        }
        Ok(())
    });
    match result {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(r) => r,
    }
}

async fn remove(State(s): State<Settings>, Path(id): Path<String>) -> Response {
    let result = s.edit(|ps| {
        let before = ps.len();
        ps.retain(|q| id_of(q) != id);
        if ps.len() == before {
            return Err(err(StatusCode::NOT_FOUND, format!("no provider '{id}'")));
        }
        Ok(())
    });
    if result.is_ok() {
        // Best effort: a key left behind in the keychain is harmless but untidy.
        let _ = keyring::Entry::new(KEYCHAIN_SERVICE, &id).and_then(|e| e.delete_credential());
    }
    match result {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(r) => r,
    }
}

// ----------------------------------------------------------------- keys

/// `{key, store: "keychain" | "file"}`. "file" exists for machines with no
/// keychain, and only because the page asked the user first.
async fn set_key(State(s): State<Settings>, Path(id): Path<String>, Json(body): Json<Value>) -> Response {
    let key = body.get("key").and_then(Value::as_str).unwrap_or("").trim().to_owned();
    if key.is_empty() {
        return err(StatusCode::BAD_REQUEST, "the key is empty");
    }
    let to_file = body.get("store").and_then(Value::as_str) == Some("file");
    if !to_file {
        let account = id.clone();
        let stored = tokio::task::spawn_blocking(move || {
            keyring::Entry::new(KEYCHAIN_SERVICE, &account).and_then(|e| e.set_password(&key))
        })
        .await;
        match stored {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return err(StatusCode::SERVICE_UNAVAILABLE, format!("{} would not store the key: {e}", keychain_name()))
            }
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
        let result = s.edit(|ps| {
            let p = ps.iter_mut().find(|q| id_of(q) == id).ok_or_else(|| err(StatusCode::NOT_FOUND, format!("no provider '{id}'")))?;
            let o = p.as_object_mut().unwrap();
            o.insert("keychain".into(), json!(true));
            o.remove("api_key"); // the keychain copy replaces any plaintext one
            Ok(())
        });
        return match result {
            Ok(()) => Json(json!({ "ok": true, "stored_in": keychain_name() })).into_response(),
            Err(r) => r,
        };
    }
    let result = s.edit(|ps| {
        let p = ps.iter_mut().find(|q| id_of(q) == id).ok_or_else(|| err(StatusCode::NOT_FOUND, format!("no provider '{id}'")))?;
        p.as_object_mut().unwrap().insert("api_key".into(), json!(key));
        Ok(())
    });
    match result {
        Ok(()) => Json(json!({ "ok": true, "stored_in": "the settings file" })).into_response(),
        Err(r) => r,
    }
}

async fn delete_key(State(s): State<Settings>, Path(id): Path<String>) -> Response {
    let account = id.clone();
    let _ = tokio::task::spawn_blocking(move || {
        keyring::Entry::new(KEYCHAIN_SERVICE, &account).and_then(|e| e.delete_credential())
    })
    .await;
    let result = s.edit(|ps| {
        if let Some(o) = ps.iter_mut().find(|q| id_of(q) == id).and_then(Value::as_object_mut) {
            o.remove("api_key");
        }
        Ok(())
    });
    match result {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(r) => r,
    }
}

// ----------------------------------------------------------------- test

/// `{base, key?, id?, headers?}` -> can we reach it, does it accept the key,
/// and which models does it list. A key in the body is one being typed and not
/// yet saved; otherwise the stored key for `id` is used.
///
/// Asks only `{base}/v1/models`. Never probe a path under a gateway's
/// /upstream/: on llama-swap that *starts* the service and can evict whatever
/// model is resident on the GPU.
async fn test(State(s): State<Settings>, Json(body): Json<Value>) -> Response {
    let base = body.get("base").and_then(Value::as_str).unwrap_or("").trim_end_matches('/').to_owned();
    if reqwest::Url::parse(&base).is_err() {
        return Json(json!({ "ok": false, "message": "That is not a web address. It should start with http:// or https://." })).into_response();
    }
    let mut key = body.get("key").and_then(Value::as_str).filter(|k| !k.is_empty()).map(str::to_owned);
    if key.is_none() {
        if let Some(id) = body.get("id").and_then(Value::as_str) {
            let stored = s.read().ok().and_then(|doc| {
                doc.get("providers")?.as_array()?.iter().find(|p| id_of(p) == id).cloned()
            });
            if let Some(p) = stored {
                key = p.get("api_key").and_then(Value::as_str).filter(|k| !k.is_empty()).map(str::to_owned);
                if key.is_none() && p.get("keychain").and_then(Value::as_bool).unwrap_or(false) {
                    let id = id.to_owned();
                    key = tokio::task::spawn_blocking(move || crate::broker::keychain_get(&id)).await.ok().flatten();
                }
            }
        }
    }
    // Workers AI has no /v1/models; its model search proves both the account id
    // in the address and the token, which is all a test needs to show.
    let cloudflare = body.get("kind").and_then(Value::as_str) == Some("cloudflare");
    if cloudflare && base.contains("ACCOUNT_ID") {
        return Json(json!({ "ok": false, "message": "Replace ACCOUNT_ID in the address with your Cloudflare account id." })).into_response();
    }
    let probe = if cloudflare { format!("{base}/models/search?search=whisper") } else { format!("{base}/v1/models") };
    let mut req = s.http.get(probe).timeout(Duration::from_secs(10));
    if let Some(k) = &key {
        req = req.bearer_auth(k);
    }
    if let Some(h) = body.get("headers").and_then(Value::as_object) {
        for (k, v) in h {
            if let Some(v) = v.as_str() {
                req = req.header(k, v);
            }
        }
    }
    let reply = match req.send().await {
        Err(e) if e.is_timeout() => json!({ "ok": false, "message": "The server did not answer within 10 seconds." }),
        Err(e) if e.is_connect() => json!({ "ok": false, "message": "Nothing is answering at that address. Check it, and that the server is running." }),
        Err(e) => json!({ "ok": false, "message": format!("Could not reach the server: {e}") }),
        Ok(r) => {
            let status = r.status();
            let doc: Value = r.json().await.unwrap_or(Value::Null);
            match status.as_u16() {
                200..=299 if cloudflare => {
                    let n = doc.get("result").and_then(Value::as_array).map_or(0, Vec::len);
                    json!({ "ok": true, "models": [],
                            "message": format!("Connected to Cloudflare. The account can use {n} Whisper model{}.", if n == 1 { "" } else { "s" }) })
                }
                200..=299 => {
                    let ids: Vec<Value> = doc
                        .get("data")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(|m| m.get("id").cloned()).collect())
                        .unwrap_or_default();
                    json!({ "ok": true, "models": ids,
                            "message": format!("Connected. The server lists {} model{}.", ids.len(), if ids.len() == 1 { "" } else { "s" }) })
                }
                401 | 403 if key.is_none() => json!({ "ok": false, "message": "The server wants an API key. Add one and test again." }),
                401 | 403 => json!({ "ok": false, "message": "The server refused this API key. Check that it was copied whole." }),
                404 => json!({ "ok": false, "message": "Something answered, but it is not a speech-recognition server (it has no /v1/models)." }),
                429 => json!({ "ok": false, "message": "The key works, but the account is over its rate limit or out of credit." }),
                code => json!({ "ok": false, "message": format!("The server answered with an error ({code}).") }),
            }
        }
    };
    Json(reply).into_response()
}
