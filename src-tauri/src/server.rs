// SPDX-License-Identifier: GPL-3.0-or-later
// The embedded server: static files plus the /api/* surface serve.py defines.

use std::{net::SocketAddr, path::PathBuf};

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde_json::json;
use tower_http::services::{ServeDir, ServeFile};

pub struct Roots {
    pub web: PathBuf,
    pub vendor: PathBuf,
}

impl Roots {
    /// Bundled resources in a release build; the source tree under `cargo run`.
    pub fn locate(resource_dir: Option<PathBuf>) -> Result<Self, String> {
        let mut candidates = Vec::new();
        if let Some(r) = resource_dir {
            candidates.push(Roots { web: r.join("web"), vendor: r.join("vendor") });
        }
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        candidates.push(Roots { web: src.join("web"), vendor: src.join("vendor") });

        candidates
            .into_iter()
            .find(|r| r.web.join("index.html").is_file()
                && r.vendor.join("js/worker-client.mjs").is_file())
            .ok_or_else(|| "praat-wasm not found: run scripts/fetch-praat-wasm.sh".into())
    }
}

pub async fn start(roots: Roots) -> std::io::Result<u16> {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await?;
    let port = listener.local_addr()?.port();
    let index = roots.web.join("index.html");

    let app = Router::new()
        .route("/api/health", get(|| async {
            Json(json!({ "status": "ok", "service": "praat-desktop", "providers": 0 }))
        }))
        // An empty list: the page falls back to its typed-endpoint mode until the
        // Rust broker exists (milestone 2).
        .route("/api/providers", get(|| async {
            Json(json!({ "providers": [], "configured": null }))
        }))
        .route("/api/capabilities", get(|| async {
            Json(json!({ "desktop": true, "version": env!("CARGO_PKG_VERSION") }))
        }))
        .route_service("/", ServeFile::new(&index))
        .route_service("/index.html", ServeFile::new(&index))
        .nest_service("/vendor", ServeDir::new(&roots.vendor))
        .layer(middleware::from_fn(move |req, next| only_loopback_host(req, next, port)));

    tauri::async_runtime::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            eprintln!("embedded server stopped: {e}");
        }
    });
    Ok(port)
}

/// Refuse any request whose Host is not our own loopback address. Without this a
/// web page in the user's ordinary browser could reach the server through DNS
/// rebinding, and once the broker holds API keys that would let it spend them.
async fn only_loopback_host(req: Request, next: Next, port: u16) -> Response {
    let ok = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .is_some_and(|h| h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}"));
    if ok {
        next.run(req).await
    } else {
        (StatusCode::FORBIDDEN, "forbidden").into_response()
    }
}
