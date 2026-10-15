// SPDX-License-Identifier: GPL-3.0-or-later
// The system's own Open and Save dialogs, for the menu bar and the export
// buttons. A page cannot open a file chooser except from a click (WebKit
// refuses a scripted one, which is what a menu command is), and a download
// lands wherever the webview decides, with no "Save As". So the app shows the
// dialog and moves the bytes: an opened file comes back as the response body,
// and a saved one arrives as the request body.

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Query},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;

pub fn router() -> Router {
    Router::new()
        .route("/api/open", post(open))
        // A figure at 600 dpi or an hour of annotation is well past axum's 2 MB default.
        .route("/api/save-as", post(save_as).layer(DefaultBodyLimit::max(1 << 30)))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

const AUDIO: &[&str] = &["wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "aif", "aiff", "webm", "mov"];
const ANNOTATION: &[&str] = &["TextGrid", "textgrid", "json", "lana"];

/// `POST /api/open?kind=audio|annotation`: the chosen file's bytes, its name
/// in `x-lana-name` (percent-encoded), or 204 when the dialog was cancelled.
async fn open(Query(q): Query<HashMap<String, String>>) -> Response {
    let Some(app) = crate::APP.get().cloned() else {
        return err(StatusCode::NOT_IMPLEMENTED, "No file dialog here.");
    };
    let audio = q.get("kind").map(String::as_str) != Some("annotation");
    let picked: Option<PathBuf> = tokio::task::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;
        let d = app.dialog().file();
        let d = if audio {
            d.set_title("Open a recording").add_filter("Recordings", AUDIO)
        } else {
            d.set_title("Open an annotation").add_filter("TextGrid or Lana file", ANNOTATION)
        };
        d.blocking_pick_file()
    })
    .await
    .ok()
    .flatten()
    .and_then(|p| p.into_path().ok());
    let Some(path) = picked else {
        return StatusCode::NO_CONTENT.into_response();
    };
    let bytes = match tokio::fs::read(&path).await {
        Ok(b) => b,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not read {}: {e}", path.display())),
    };
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    Response::builder()
        .header(header::CONTENT_TYPE, mime.as_ref())
        .header("x-lana-name", percent(&name))
        .body(Body::from(bytes))
        .unwrap()
}

/// `POST /api/save-as?name=<suggested>` with the file as the body: the path it
/// was written to, or `{"cancelled": true}`.
async fn save_as(Query(q): Query<HashMap<String, String>>, body: Bytes) -> Response {
    let Some(app) = crate::APP.get().cloned() else {
        return err(StatusCode::NOT_IMPLEMENTED, "No file dialog here.");
    };
    let name = q.get("name").cloned().unwrap_or_else(|| "lana-export".into());
    // The suggested name only: anything path-like from the page is dropped.
    let name = name.rsplit(['/', '\\']).next().unwrap_or("lana-export").to_string();
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_string());
    let chosen: Option<PathBuf> = tokio::task::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;
        let mut d = app.dialog().file().set_title("Save as").set_file_name(&name);
        if let Some(e) = ext.as_deref() {
            d = d.add_filter(e.to_uppercase(), &[e]);
        }
        d.blocking_save_file()
    })
    .await
    .ok()
    .flatten()
    .and_then(|p| p.into_path().ok());
    let Some(path) = chosen else {
        return Json(json!({ "cancelled": true })).into_response();
    };
    match tokio::fs::write(&path, &body).await {
        Ok(()) => Json(json!({ "path": path.display().to_string() })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not write {}: {e}", path.display())),
    }
}

/// Enough percent-encoding for a header value: everything outside visible ASCII.
fn percent(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~ ()".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}
