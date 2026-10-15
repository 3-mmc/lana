// SPDX-License-Identifier: GPL-3.0-or-later
// Updates: Lana asks GitHub Releases for latest.json at start, and the page
// offers "Update now" when a newer version exists. The download is checked
// against the public key in tauri.conf.json (the private key signs releases in
// CI), then installed, and Lana restarts.
//
// Installed this way, an update never passes through a browser, so macOS does
// not quarantine it and the "Open Anyway" step happens once, at first install.
// A .deb cannot replace itself; there the page links to the release instead.

use axum::{http::StatusCode, response::{IntoResponse, Response}, routing::{get, post}, Json, Router};
use serde_json::json;
use tauri_plugin_updater::UpdaterExt;

pub fn router() -> Router {
    Router::new()
        .route("/api/update", get(check))
        .route("/api/update/install", post(install))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, Json(json!({ "error": msg.into() }))).into_response()
}

async fn check() -> Response {
    let Some(app) = crate::APP.get() else {
        return Json(json!({ "available": false, "supported": false })).into_response();
    };
    let current = app.package_info().version.to_string();
    let updater = match app.updater() {
        Ok(u) => u,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    match updater.check().await {
        Ok(Some(u)) => Json(json!({ "available": true, "current": current, "version": u.version,
                                    "notes": u.body, "date": u.date.map(|d| d.to_string()) })).into_response(),
        Ok(None) => Json(json!({ "available": false, "current": current })).into_response(),
        // Offline, or GitHub unreachable: not something to bother the user with.
        Err(e) => Json(json!({ "available": false, "current": current, "error": e.to_string() })).into_response(),
    }
}

async fn install() -> Response {
    let Some(app) = crate::APP.get().cloned() else {
        return err(StatusCode::NOT_IMPLEMENTED, "Updates are installed by the app.");
    };
    let update = match app.updater().map_err(|e| e.to_string()) {
        Ok(u) => match u.check().await {
            Ok(Some(update)) => update,
            Ok(None) => return err(StatusCode::CONFLICT, "Lana is already up to date."),
            Err(e) => return err(StatusCode::BAD_GATEWAY, format!("Could not reach the update server: {e}")),
        },
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    match update.download_and_install(|_, _| {}, || {}).await {
        Ok(()) => {
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                app.restart();
            });
            Json(json!({ "ok": true })).into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, format!("The update could not be installed: {e}")),
    }
}
