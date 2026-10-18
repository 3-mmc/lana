// SPDX-License-Identifier: GPL-3.0-or-later
// The menu bar. On macOS every app has one at the top of the screen, and
// without Lana's own the system showed Tauri's default (no File menu at all).
// Commands are the page's: a menu item calls window.lanaMenu(id) in the page,
// which runs the same code as the button or key that does it there.
//
// Cut, Copy, Paste and Select All stay the system's own items, because on macOS
// those are what make ⌘C and ⌘V work in a text field at all. Undo and Redo are
// Lana's, so they undo annotation edits; in a text field the page hands them
// back to the field.

use tauri::menu::{AboutMetadata, Menu, MenuEvent, MenuItem, MenuItemBuilder, SubmenuBuilder};
use tauri::{AppHandle, Manager, Wry};

/// Whether this platform gets the menu bar: macOS always; elsewhere only with
/// LANA_MENU set, for testing, since a menu bar under a Windows or Linux title
/// bar would only repeat the page's own buttons.
pub fn wanted() -> bool {
    cfg!(target_os = "macos") || std::env::var_os("LANA_MENU").is_some()
}

pub fn build(app: &AppHandle) -> tauri::Result<Menu<Wry>> {
    let item = |id: &str, label: &str, accel: Option<&str>| -> tauri::Result<MenuItem<Wry>> {
        let mut b = MenuItemBuilder::with_id(id, label);
        if let Some(a) = accel {
            b = b.accelerator(a);
        }
        b.build(app)
    };

    let about = AboutMetadata {
        name: Some("Lana".into()),
        version: Some(app.package_info().version.to_string()),
        comments: Some("Linguistic ANAlysis. Built on Praat; not affiliated with the Praat project.".into()),
        website: Some("https://openlana.app".into()),
        website_label: Some("openlana.app".into()),
        license: Some("GPL-3.0-or-later".into()),
        authors: Some(vec!["Aaron van Blerkom".into()]),
        copyright: Some("© 2026 Aaron van Blerkom".into()),
        ..Default::default()
    };
    let app_menu = SubmenuBuilder::new(app, "Lana")
        .about(Some(about))
        .separator()
        .item(&item("settings", "Recognisers…", Some("CmdOrCtrl+,"))?)
        .item(&item("check-update", "Check for Updates…", None)?)
        .separator()
        .services()
        .separator()
        .hide()
        .hide_others()
        .show_all()
        .separator()
        .quit()
        .build()?;

    let export = SubmenuBuilder::new(app, "Export")
        .item(&item("export-textgrid", "TextGrid…", Some("CmdOrCtrl+E"))?)
        .item(&item("export-srt", "Subtitles (SRT)…", None)?)
        .item(&item("export-vtt", "Subtitles (VTT)…", None)?)
        .item(&item("export-ass", "Subtitles (ASS)…", None)?)
        .item(&item("export-tsv", "Table (TSV)…", None)?)
        .separator()
        .item(&item("figure", "Figure / Measurements…", None)?)
        .build()?;
    let file = SubmenuBuilder::new(app, "File")
        .item(&item("open-audio", "Open Recording…", Some("CmdOrCtrl+O"))?)
        .item(&item("open-annotation", "Open Annotation…", Some("CmdOrCtrl+Shift+O"))?)
        .item(&item("record", "Record…", None)?)
        .separator()
        .item(&item("save-lana", "Save Annotation As…", Some("CmdOrCtrl+S"))?)
        .item(&export)
        .separator()
        .close_window()
        .build()?;

    let edit = SubmenuBuilder::new(app, "Edit")
        .item(&item("undo", "Undo", Some("CmdOrCtrl+Z"))?)
        .item(&item("redo", "Redo", Some("CmdOrCtrl+Shift+Z"))?)
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .separator()
        .item(&item("add-tier", "Add Tier…", None)?)
        // N and F in the page; no accelerator, since a bare letter would be
        // taken from every text field.
        .item(&item("add-note", "Add Note", None)?)
        .item(&item("flag", "Flag Interval for Later", None)?)
        .build()?;

    let view = SubmenuBuilder::new(app, "View")
        .item(&item("play", "Play / Pause", None)?)
        .item(&item("stop-playback", "Stop Playback", None)?)
        .item(&item("toggle-follow", "Follow Playback", None)?)
        .separator()
        .item(&item("prev-note", "Previous Note", None)?)
        .item(&item("next-note", "Next Note", None)?)
        .separator()
        .item(&item("zoom-sel", "Zoom to Selection", Some("CmdOrCtrl+Shift+E"))?)
        .item(&item("zoom-in", "Zoom In", Some("CmdOrCtrl+="))?)
        .item(&item("zoom-out", "Zoom Out", Some("CmdOrCtrl+-"))?)
        .item(&item("zoom-all", "Show the Whole Recording", Some("CmdOrCtrl+0"))?)
        .separator()
        .item(&item("toggle-f0", "Show or Hide f0", None)?)
        .item(&item("toggle-formants", "Show or Hide Formants", None)?)
        .separator()
        .item(&item("theme-graphite", "Graphite Theme", None)?)
        .item(&item("theme-paper", "Paper Theme", None)?)
        .separator()
        .fullscreen()
        .build()?;

    let window = SubmenuBuilder::new(app, "Window").minimize().maximize().separator().close_window().build()?;

    let help = SubmenuBuilder::new(app, "Help")
        .item(&item("help-site", "Lana Website", None)?)
        .item(&item("help-guide", "Lana on GitHub", None)?)
        .item(&item("help-issues", "Report a Problem…", None)?)
        .build()?;

    Menu::with_items(app, &[&app_menu, &file, &edit, &view, &window, &help])
}

pub fn on_event(app: &AppHandle, event: MenuEvent) {
    let id = event.id().0.as_str();
    if let Some(w) = app.get_webview_window("main") {
        let id = serde_json::to_string(id).unwrap_or_default();
        let _ = w.eval(format!("window.lanaMenu && window.lanaMenu({id})"));
    }
}
