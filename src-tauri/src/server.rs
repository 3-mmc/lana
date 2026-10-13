// SPDX-License-Identifier: GPL-3.0-or-later
// The embedded server: static files plus the /api/* surface serve.py defines.

use std::{net::SocketAddr, path::PathBuf};

use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    Router,
};
use tower_http::services::{ServeDir, ServeFile};

pub struct Roots {
    pub web: PathBuf,
    pub vendor: PathBuf,
    /// providers.json; need not exist (no providers, and the page falls back
    /// to its typed-endpoint mode).
    pub providers: PathBuf,
    /// A recording to open at start, from the command line or the OS.
    pub launch: Option<PathBuf>,
}

impl Roots {
    /// Bundled resources in a release build; the source tree under `cargo run`.
    pub fn locate(resource_dir: Option<PathBuf>, providers: PathBuf) -> Result<Self, String> {
        let mut candidates = Vec::new();
        if let Some(r) = resource_dir {
            candidates.push(Roots { web: r.join("web"), vendor: r.join("vendor"), providers: providers.clone(), launch: None });
        }
        let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
        candidates.push(Roots { web: src.join("web"), vendor: src.join("vendor"), providers, launch: None });

        candidates
            .into_iter()
            .find(|r| r.web.join("index.html").is_file()
                && r.vendor.join("js/worker-client.mjs").is_file())
            .ok_or_else(|| "praat-wasm not found: run scripts/fetch-praat-wasm.sh".into())
    }
}

pub async fn start(roots: Roots) -> std::io::Result<u16> {
    start_on(roots, 0).await
}

/// Port 0 picks a free one, which is what the app wants.
pub async fn start_on(roots: Roots, port: u16) -> std::io::Result<u16> {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], port))).await?;
    let port = listener.local_addr()?.port();
    let index = roots.web.join("index.html");

    let app = Router::new()
        .merge(crate::broker::router(roots.providers.clone()))
        .merge(crate::settings::router(roots.providers.clone(), roots.launch.clone()))
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

/// Only our own page may talk to this server.
///
/// The Host check stops DNS rebinding. It does not stop a page in the user's
/// ordinary browser from POSTing straight to 127.0.0.1:<port>: a multipart form
/// is a "simple" request, sent without a CORS preflight, and although the page
/// could not read the answer it could make the broker spend the user's API key.
/// Browsers attach Origin to every cross-origin fetch and form POST, so a
/// foreign Origin is refused too. A request with no Origin is a same-origin GET
/// or a non-browser client, which has no ambient authority to abuse.
async fn only_loopback_host(req: Request, next: Next, port: u16) -> Response {
    let ours = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    let allowed = {
        let get = |name: header::HeaderName| req.headers().get(name).and_then(|h| h.to_str().ok());
        let host_ok = get(header::HOST).is_some_and(|h| ours.iter().any(|o| o == h));
        let origin_ok = get(header::ORIGIN)
            .is_none_or(|o| ours.iter().any(|h| o == format!("http://{h}")));
        host_ok && origin_ok
    };
    if allowed {
        next.run(req).await
    } else {
        (StatusCode::FORBIDDEN, "forbidden").into_response()
    }
}
