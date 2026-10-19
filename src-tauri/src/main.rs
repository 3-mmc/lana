// SPDX-License-Identifier: GPL-3.0-or-later
// Lana: lana-web's interface in a native window.
//
// The page is served by an embedded HTTP server on 127.0.0.1 rather than through
// Tauri's asset protocol, so workers, wasm and same-origin fetches behave exactly
// as they do in the browser build, and the page's /api/* contract (serve.py's)
// can be answered from Rust without the page knowing it is not in a browser.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod broker;
mod local;
mod recordings;
mod store;
mod server;
mod settings;
mod models_dir;
mod update;
mod files;
mod menu;

/// The running app, for the few server routes that need the native side: the
/// folder picker and a restart.
pub static APP: std::sync::OnceLock<tauri::AppHandle> = std::sync::OnceLock::new();

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
        let mut roots = server::Roots::locate(None, providers.into()).expect("source tree");
        roots.models = std::env::var_os("LANA_MODELS").map(Into::into).unwrap_or_else(|| std::env::temp_dir().join("lana-models"));
        roots.models_pointer = std::env::var_os("LANA_MODELS_POINTER").map(Into::into);
        roots.recordings = std::env::var_os("LANA_RECORDINGS").map(Into::into).unwrap_or_else(|| std::env::temp_dir().join("lana-recordings"));
        roots.store = std::env::var_os("LANA_STORE").map(Into::into).unwrap_or_else(|| std::env::temp_dir().join("lana-store"));
        tauri::async_runtime::block_on(async move {
            let port = server::start_on(roots, port).await.expect("bind");
            eprintln!("lana-desktop serving http://127.0.0.1:{port}/ (no window)");
            std::future::pending::<()>().await
        });
        return;
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let _ = APP.set(app.handle().clone());
            // LANA_PROVIDERS overrides, so a development run can share serve.py's file.
            let providers = match std::env::var_os("LANA_PROVIDERS") {
                Some(p) => p.into(),
                None => app.path().app_config_dir()?.join("providers.json"),
            };
            eprintln!("providers from {}", providers.display());
            let mut roots = server::Roots::locate(app.path().resource_dir().ok(), providers)?;
            // The ONNX Runtime library ships as a resource (local/runtime.rs).
            if let Ok(r) = app.path().resource_dir() {
                local::runtime::look_in(r.join("onnxruntime"));
            }
            // `lana-desktop recording.wav`: the first argument that names a file.
            // (macOS "Open with" arrives as an event instead; not handled yet.)
            // A folder the user chose in the settings screen, recorded in the
            // config folder; the data folder otherwise.
            let pointer = app.path().app_config_dir()?.join("models-location.txt");
            roots.models = match std::env::var_os("LANA_MODELS") {
                Some(p) => p.into(),
                None => models_dir::chosen(&pointer).unwrap_or(app.path().app_data_dir()?.join("models")),
            };
            roots.models_pointer = Some(pointer);
            // Recordings go where a user will look for them, not into app data.
            // document_dir() fails where no XDG Documents folder is configured
            // (minimal Linux setups, WSL). That must not stop the app starting.
            roots.recordings = match std::env::var_os("LANA_RECORDINGS") {
                Some(p) => p.into(),
                None => app
                    .path()
                    .document_dir()
                    .or_else(|_| app.path().home_dir().map(|h| h.join("Documents")))
                    .or_else(|_| app.path().app_data_dir())?
                    .join("Lana Recordings"),
            };
            roots.store = match std::env::var_os("LANA_STORE") {
                Some(p) => p.into(),
                None => app.path().app_data_dir()?.join("store"),
            };
            roots.launch = std::env::args_os().skip(1).map(std::path::PathBuf::from).find(|p| p.is_file());
            let port = tauri::async_runtime::block_on(server::start(roots))?;
            let url = format!("http://127.0.0.1:{port}/");
            eprintln!("lana-desktop serving {url}");
            let url = url.parse()?;
            let window = WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("Lana")
                .inner_size(1400.0, 900.0)
                .min_inner_size(900.0, 600.0)
                // Let the page receive dropped files as HTML5 drops. Tauri's own
                // handler would otherwise consume them before the page sees them.
                .disable_drag_drop_handler()
                .build()?;
            if menu::wanted() {
                app.set_menu(menu::build(app.handle())?)?;
                app.on_menu_event(menu::on_event);
            }
            #[cfg(target_os = "linux")]
            allow_microphone(&window);
            #[cfg(not(target_os = "linux"))]
            let _ = window;
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Lana");
}

/// WebKitGTK refuses getUserMedia unless media streams are enabled and the app
/// answers the permission request; with no handler the answer is no. Grant
/// microphone requests (never camera) and the device-name list, nothing else.
/// Only this window asks, and it shows nothing but Lana's own page.
#[cfg(target_os = "linux")]
fn allow_microphone(window: &tauri::WebviewWindow) {
    let _ = window.with_webview(|pv| {
        use webkit2gtk::glib::prelude::Cast;
        use webkit2gtk::{
            DeviceInfoPermissionRequest, PermissionRequestExt, SettingsExt, UserMediaPermissionRequest,
            UserMediaPermissionRequestExt, WebViewExt,
        };
        let wv = pv.inner();
        if let Some(settings) = WebViewExt::settings(&wv) {
            settings.set_enable_media_stream(true);
        }
        wv.connect_permission_request(|_, req| {
            let microphone = req
                .downcast_ref::<UserMediaPermissionRequest>()
                .is_some_and(|um| um.is_for_audio_device() && !um.is_for_video_device());
            // Device names, so the input list says "USB microphone", not "microphone".
            let device_names = req.downcast_ref::<DeviceInfoPermissionRequest>().is_some();
            if microphone || device_names {
                req.allow();
                return true;
            }
            false
        });
    });
}
