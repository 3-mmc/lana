// SPDX-License-Identifier: GPL-3.0-or-later
// Small JSON documents the page keeps between sessions: the annotation of each
// recording, and the page's own settings.
//
// The page cannot keep them itself. The server listens on a new port every
// launch, a port is part of the page's origin, and browser storage belongs to
// an origin, so anything the page put in localStorage or IndexedDB was out of
// reach the next time Lana started. Here they are files in the app's data
// folder, one per document, written atomically.

use std::path::PathBuf;

use axum::{
    body::Bytes,
    extract::{DefaultBodyLimit, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::{json, Value};

/// What the page may store. Anything else is refused, so the endpoint cannot
/// be used to write files of arbitrary kinds into the data folder.
const SPACES: &[&str] = &["annotations", "prefs"];
const MAX_BYTES: usize = 32 * 1024 * 1024;

pub fn router(dir: PathBuf) -> Router {
    Router::new()
        .route("/api/store/{space}/{key}", get(read).put(write).delete(remove))
        // axum's default limit is 2 MB, which a long recording's phone tier
        // with its timings can pass.
        .layer(DefaultBodyLimit::max(MAX_BYTES))
        .with_state(dir)
}

fn err(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({ "error": msg }))).into_response()
}

/// `<dir>/<space>/<key>.json`, or None for a space or key that is not allowed.
/// Keys are what the page derives from a recording's contents (hex digits), so
/// letters, digits, `-` and `_` cover them, and nothing that could name a path.
fn path(dir: &std::path::Path, space: &str, key: &str) -> Option<PathBuf> {
    let ok_key = !key.is_empty() && key.len() <= 80 && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    (SPACES.contains(&space) && ok_key).then(|| dir.join(space).join(format!("{key}.json")))
}

async fn read(State(dir): State<PathBuf>, Path((space, key)): Path<(String, String)>) -> Response {
    let Some(p) = path(&dir, &space, &key) else { return err(StatusCode::BAD_REQUEST, "bad name") };
    match tokio::fs::read(&p).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, "application/json")], bytes).into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => err(StatusCode::NOT_FOUND, "nothing saved"),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn write(State(dir): State<PathBuf>, Path((space, key)): Path<(String, String)>, body: Bytes) -> Response {
    let Some(p) = path(&dir, &space, &key) else { return err(StatusCode::BAD_REQUEST, "bad name") };
    if body.len() > MAX_BYTES {
        return err(StatusCode::PAYLOAD_TOO_LARGE, "too large to save");
    }
    if serde_json::from_slice::<Value>(&body).is_err() {
        return err(StatusCode::BAD_REQUEST, "not JSON");
    }
    let result = async {
        tokio::fs::create_dir_all(p.parent().unwrap()).await?;
        // Write beside the target and rename over it, so a crash mid-write
        // leaves the previous save rather than half of this one.
        let tmp = p.with_extension("json.tmp");
        tokio::fs::write(&tmp, &body).await?;
        tokio::fs::rename(&tmp, &p).await
    }
    .await;
    match result {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) if e.raw_os_error() == Some(28) || e.raw_os_error() == Some(112) => {
            err(StatusCode::INSUFFICIENT_STORAGE, "The disk is full.")
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

async fn remove(State(dir): State<PathBuf>, Path((space, key)): Path<(String, String)>) -> Response {
    let Some(p) = path(&dir, &space, &key) else { return err(StatusCode::BAD_REQUEST, "bad name") };
    match tokio::fs::remove_file(&p).await {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_known_spaces_and_plain_keys() {
        let d = std::path::Path::new("/data");
        assert!(path(d, "annotations", "0a1b2c3d").is_some());
        assert!(path(d, "prefs", "config").is_some());
        assert!(path(d, "models", "x").is_none());
        assert!(path(d, "annotations", "../x").is_none());
        assert!(path(d, "annotations", "a/b").is_none());
        assert!(path(d, "annotations", "").is_none());
    }
}
