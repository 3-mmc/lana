#[path = "../scripts/macos-link.rs"]
mod macos_link;

fn main() {
    macos_link::configure();
    tauri_build::build()
}
