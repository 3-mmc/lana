// SPDX-License-Identifier: GPL-3.0-or-later
// Where downloaded models live, and moving them somewhere else.
//
// Models take gigabytes (MMS alone is 1 GB), and the app's data folder is on
// the system disk, which on many laptops is the small one. The settings screen
// shows the folder, opens it in the file manager, and moves the models to a
// folder the user picks. The choice is a path in models-location.txt in the
// config folder, read once at start, so a move takes effect after a restart.

use std::path::{Path, PathBuf};

use axum::{extract::State, http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}, Json, Router};
use serde_json::{json, Value};

#[derive(Clone)]
struct Dir {
    current: PathBuf,
    pointer: Option<PathBuf>,
    local: crate::local::Local,
}

/// The folder recorded by an earlier choice, if it is still usable.
pub fn chosen(pointer: &Path) -> Option<PathBuf> {
    let p = PathBuf::from(std::fs::read_to_string(pointer).ok()?.trim());
    (p.is_absolute() && p.is_dir()).then_some(p)
}

pub fn router(current: PathBuf, pointer: Option<PathBuf>, local: crate::local::Local) -> Router {
    Router::new()
        .route("/api/settings/models-dir", get(show).post(change))
        .route("/api/settings/models-dir/open", post(open))
        .route("/api/settings/models-dir/pick", post(pick))
        .route("/api/restart", post(restart))
        .with_state(Dir { current, pointer, local })
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

/// Large files with the same name and size are the same file: each MMS
/// language holds mms-base.bin (1 GB) as a hard link to the shared copy, and
/// counting or copying it once per language would multiply it.
type Seen = std::collections::HashMap<(std::ffi::OsString, u64), PathBuf>;
const SHARED_MIN: u64 = 1 << 20;

/// Bytes on disk, counting each shared large file once.
fn size_of(p: &Path) -> u64 {
    fn walk(p: &Path, seen: &mut Seen) -> u64 {
        std::fs::read_dir(p).map_or(0, |rd| rd.flatten().map(|e| match e.metadata() {
            Ok(m) if m.is_dir() => walk(&e.path(), seen),
            Ok(m) if m.len() >= SHARED_MIN => {
                if seen.insert((e.file_name(), m.len()), e.path()).is_some() { 0 } else { m.len() }
            }
            Ok(m) => m.len(),
            _ => 0,
        }).sum())
    }
    walk(p, &mut Seen::new())
}

async fn show(State(d): State<Dir>) -> Json<Value> {
    let current = d.current.clone();
    let bytes = tokio::task::spawn_blocking(move || size_of(&current)).await.unwrap_or(0);
    // The folder that will be used after a restart, if a move is pending.
    let next = d.pointer.as_deref().and_then(chosen).filter(|p| p != &d.current);
    Json(json!({
        "dir": d.current.display().to_string(), "bytes": bytes,
        "can_change": d.pointer.is_some(), "pending": next.map(|p| p.display().to_string()),
    }))
}

/// Show the folder in Finder, Explorer or the desktop's file manager.
async fn open(State(d): State<Dir>) -> Response {
    let _ = std::fs::create_dir_all(&d.current);
    let cmd = if cfg!(target_os = "macos") { "open" } else if cfg!(windows) { "explorer" } else { "xdg-open" };
    match std::process::Command::new(cmd).arg(&d.current).spawn() {
        Ok(_) => Json(json!({ "ok": true })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not open the folder: {e}")),
    }
}

/// The system's own folder picker. The page cannot open one that returns a
/// path, so the app opens it and hands the path back.
async fn pick() -> Response {
    let Some(app) = crate::APP.get().cloned() else {
        return err(StatusCode::NOT_IMPLEMENTED, "No folder picker here.");
    };
    let picked = tokio::task::spawn_blocking(move || {
        use tauri_plugin_dialog::DialogExt;
        app.dialog().file().set_title("Where should Lana keep its models?").blocking_pick_folder()
    })
    .await
    .ok()
    .flatten()
    .and_then(|p| p.into_path().ok());
    Json(json!({ "path": picked.map(|p| p.display().to_string()) })).into_response()
}

/// Move the models to `path` and record it for the next start. The models are
/// moved, not copied, so they never occupy both disks; a rename where the two
/// folders share a disk, a copy and delete where they do not.
async fn change(State(d): State<Dir>, Json(body): Json<Value>) -> Response {
    let Some(pointer) = d.pointer.clone() else {
        return err(StatusCode::NOT_IMPLEMENTED, "The models folder cannot be changed here.");
    };
    let Some(to) = body["path"].as_str().map(str::trim).filter(|p| !p.is_empty()).map(PathBuf::from) else {
        return err(StatusCode::BAD_REQUEST, "Choose a folder.");
    };
    if !to.is_absolute() {
        return err(StatusCode::BAD_REQUEST, "Choose a full folder path.");
    }
    if d.local.busy() {
        return err(StatusCode::CONFLICT, "A model is downloading or running. Try again when it has finished.");
    }
    let from = d.current.clone();
    if to.starts_with(&from) {
        return err(StatusCode::BAD_REQUEST, "That folder is inside the current models folder.");
    }
    let result = tokio::task::spawn_blocking(move || -> Result<u64, String> {
        std::fs::create_dir_all(&to).map_err(|e| format!("Cannot use that folder: {e}"))?;
        let probe = to.join(".lana-write-test");
        std::fs::write(&probe, b"").map_err(|e| format!("Lana cannot write to that folder: {e}"))?;
        let _ = std::fs::remove_file(&probe);
        let moved = size_of(&from);
        let mut seen = Seen::new();
        for e in std::fs::read_dir(&from).into_iter().flatten().flatten() {
            let (src, dst) = (e.path(), to.join(e.file_name()));
            if dst.exists() {
                return Err(format!("{} already exists in that folder.", dst.display()));
            }
            if std::fs::rename(&src, &dst).is_err() {
                copy_all(&src, &dst, &mut seen).map_err(|e| format!("Moving {} failed: {e}", src.display()))?;
                if src.is_dir() { std::fs::remove_dir_all(&src) } else { std::fs::remove_file(&src) }
                    .map_err(|e| format!("Copied, but could not remove {}: {e}", src.display()))?;
            }
        }
        if let Some(dir) = pointer.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        std::fs::write(&pointer, to.display().to_string()).map_err(|e| e.to_string())?;
        Ok(moved)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    match result {
        Ok(bytes) => Json(json!({ "ok": true, "moved_bytes": bytes, "restart": true })).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

/// Copy a file or folder to another disk. A large file already copied under
/// the same name and size is hard-linked to that copy instead, so the shared
/// MMS weights arrive once, as they left.
fn copy_all(src: &Path, dst: &Path, seen: &mut Seen) -> std::io::Result<()> {
    if src.is_dir() {
        std::fs::create_dir_all(dst)?;
        for e in std::fs::read_dir(src)?.flatten() {
            copy_all(&e.path(), &dst.join(e.file_name()), seen)?;
        }
        return Ok(());
    }
    let len = std::fs::metadata(src)?.len();
    let key = (src.file_name().unwrap_or_default().to_owned(), len);
    if len >= SHARED_MIN {
        if let Some(first) = seen.get(&key) {
            if std::fs::hard_link(first, dst).is_ok() {
                return Ok(());
            }
        }
    }
    std::fs::copy(src, dst)?;
    if len >= SHARED_MIN {
        seen.entry(key).or_insert_with(|| dst.to_path_buf());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_copy_keeps_shared_weights_shared() {
        let root = std::env::temp_dir().join(format!("lana-mdir-{}", std::process::id()));
        let (a, b) = (root.join("a"), root.join("b"));
        std::fs::create_dir_all(a.join("mms/dar")).unwrap();
        std::fs::write(a.join("mms/mms-base.bin"), vec![7u8; 2 << 20]).unwrap();
        std::fs::hard_link(a.join("mms/mms-base.bin"), a.join("mms/dar/mms-base.bin")).unwrap();
        std::fs::write(a.join("mms/dar/tokens.txt"), b"a 1").unwrap();
        assert_eq!(size_of(&a), (2 << 20) + 3);
        copy_all(&a, &b, &mut Seen::new()).unwrap();
        assert_eq!(size_of(&b), (2 << 20) + 3);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(b.join("mms/mms-base.bin")).unwrap().ino(),
                       std::fs::metadata(b.join("mms/dar/mms-base.bin")).unwrap().ino());
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}

async fn restart() -> Response {
    match crate::APP.get() {
        Some(app) => {
            let app = app.clone();
            // Answer first, then restart, so the page is not left waiting.
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                app.restart();
            });
            Json(json!({ "ok": true })).into_response()
        }
        None => err(StatusCode::NOT_IMPLEMENTED, "Restart Lana yourself to use the new folder."),
    }
}
