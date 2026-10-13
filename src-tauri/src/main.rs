// SPDX-License-Identifier: GPL-3.0-or-later
// Lana: lana-web's interface in a native window.
//
// The page is served by an embedded HTTP server on 127.0.0.1 rather than through
// Tauri's asset protocol, so workers, wasm and same-origin fetches behave exactly
// as they do in the browser build, and the page's /api/* contract (serve.py's)
// can be answered from Rust without the page knowing it is not in a browser.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod broker;
mod server;

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

fn main() {
    // WebKitGTK's DMA-BUF renderer draws a blank window on many Linux GPU stacks
    // (WSLg among them). Respect an explicit setting either way.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    // LANA_SERVE_ONLY=<port>: the embedded server with no window, from the source
    // tree. For scripts/parity-test.py and CI; not something a user sees.
    if let Some(port) = std::env::var_os("LANA_SERVE_ONLY") {
        let port: u16 = port.to_str().and_then(|p| p.parse().ok()).expect("LANA_SERVE_ONLY=<port>");
        let providers = std::env::var_os("LANA_PROVIDERS").expect("LANA_PROVIDERS is required here");
        let roots = server::Roots::locate(None, providers.into()).expect("source tree");
        tauri::async_runtime::block_on(async move {
            let port = server::start_on(roots, port).await.expect("bind");
            eprintln!("lana-desktop serving http://127.0.0.1:{port}/ (no window)");
            std::future::pending::<()>().await
        });
        return;
    }

    tauri::Builder::default()
        .setup(|app| {
            // LANA_PROVIDERS overrides, so a development run can share serve.py's file.
            let providers = match std::env::var_os("LANA_PROVIDERS") {
                Some(p) => p.into(),
                None => app.path().app_config_dir()?.join("providers.json"),
            };
            eprintln!("providers from {}", providers.display());
            let roots = server::Roots::locate(app.path().resource_dir().ok(), providers)?;
            let port = tauri::async_runtime::block_on(server::start(roots))?;
            let url = format!("http://127.0.0.1:{port}/");
            eprintln!("lana-desktop serving {url}");
            let url = url.parse()?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Lana")
                .inner_size(1400.0, 900.0)
                .min_inner_size(900.0, 600.0)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Lana");
}
