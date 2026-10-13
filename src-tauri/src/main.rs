// SPDX-License-Identifier: GPL-3.0-or-later
// Praat Desktop: praat-web's interface in a native window.
//
// The page is served by an embedded HTTP server on 127.0.0.1 rather than through
// Tauri's asset protocol, so workers, wasm and same-origin fetches behave exactly
// as they do in the browser build, and the page's /api/* contract (serve.py's)
// can be answered from Rust without the page knowing it is not in a browser.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod server;

use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

fn main() {
    // WebKitGTK's DMA-BUF renderer draws a blank window on many Linux GPU stacks
    // (WSLg among them). Respect an explicit setting either way.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    tauri::Builder::default()
        .setup(|app| {
            let roots = server::Roots::locate(app.path().resource_dir().ok())?;
            let port = tauri::async_runtime::block_on(server::start(roots))?;
            let url = format!("http://127.0.0.1:{port}/");
            eprintln!("praat-desktop serving {url}");
            let url = url.parse()?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Praat Desktop")
                .inner_size(1400.0, 900.0)
                .min_inner_size(900.0, 600.0)
                .build()?;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Praat Desktop");
}
