// SPDX-License-Identifier: GPL-3.0-or-later
// The catalogue of models Lana can download and run itself, and the downloads.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

const HF_REPO: &str = "ggerganov/whisper.cpp";

pub struct Entry {
    pub id: &'static str,
    pub file: &'static str,
    pub label: &'static str,
    pub note: &'static str,
    pub size_mb: u32,
    /// whisper.cpp's alignment-head preset, which DTW word timing needs.
    pub dtw: whisper_rs::DtwModelPreset,
}

/// Short and opinionated. The first one fits nearly every computer; the last
/// one is the best of these and still under 600 MB thanks to quantisation.
pub fn catalogue() -> Vec<Entry> {
    use whisper_rs::DtwModelPreset as P;
    vec![
        Entry { id: "whisper-base", file: "ggml-base.bin", size_mb: 142, dtw: P::Base,
                label: "Whisper base",
                note: "Small and quick. Good enough to try Lana out; misses more words in noisy or fast speech." },
        Entry { id: "whisper-small", file: "ggml-small.bin", size_mb: 466, dtw: P::Small,
                label: "Whisper small",
                note: "A good balance on an ordinary laptop." },
        Entry { id: "whisper-large-v3-turbo-q5", file: "ggml-large-v3-turbo-q5_0.bin", size_mb: 547, dtw: P::LargeV3Turbo,
                label: "Whisper large-v3 turbo",
                note: "The most accurate here, especially for languages other than English. Slower without a fast processor." },
    ]
}

pub fn find(id: &str) -> Option<Entry> {
    catalogue().into_iter().find(|e| e.id == id)
}

#[derive(Clone, Default)]
pub struct Download {
    pub done: u64,
    pub total: u64,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct Models {
    pub dir: PathBuf,
    downloads: Arc<Mutex<HashMap<String, Download>>>,
    http: reqwest::Client,
}

impl Models {
    pub fn new(dir: PathBuf) -> Self {
        Models {
            dir,
            downloads: Default::default(),
            http: reqwest::Client::builder()
                .user_agent(concat!("lana-desktop/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("HTTP client"),
        }
    }

    pub fn path(&self, e: &Entry) -> PathBuf {
        self.dir.join(e.file)
    }

    pub fn installed(&self) -> Vec<Entry> {
        catalogue().into_iter().filter(|e| self.path(e).is_file()).collect()
    }

    /// The catalogue as the settings screen shows it.
    pub fn list(&self) -> Value {
        let downloads = self.downloads.lock().unwrap().clone();
        let entries: Vec<Value> = catalogue()
            .iter()
            .map(|e| {
                let d = downloads.get(e.id);
                json!({
                    "id": e.id, "label": e.label, "note": e.note, "size_mb": e.size_mb,
                    "installed": self.path(e).is_file(),
                    "downloading": d.filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
                    "error": d.and_then(|d| d.error.clone()),
                })
            })
            .collect();
        json!({ "models": entries, "dir": self.dir.display().to_string() })
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let e = find(id).ok_or("no such model")?;
        let _ = std::fs::remove_file(self.path(&e).with_extension("part"));
        match std::fs::remove_file(self.path(&e)) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(err) => Err(err.to_string()),
        }
    }

    /// Start a download in the background. Returns at once; poll `list()`.
    pub fn start(&self, id: &str) -> Result<(), String> {
        let e = find(id).ok_or("no such model")?;
        {
            let mut dl = self.downloads.lock().unwrap();
            if dl.get(id).is_some_and(|d| d.error.is_none()) {
                return Ok(()); // already running
            }
            dl.insert(id.to_owned(), Download::default());
        }
        let me = self.clone();
        let id = id.to_owned();
        tauri::async_runtime::spawn(async move {
            let result = me.fetch(&e).await;
            let mut dl = me.downloads.lock().unwrap();
            match result {
                Ok(()) => {
                    dl.remove(&id);
                }
                Err(msg) => {
                    dl.insert(id, Download { error: Some(msg), ..Default::default() });
                }
            }
        });
        Ok(())
    }

    fn report(&self, id: &str, done: u64, total: u64) {
        if let Some(d) = self.downloads.lock().unwrap().get_mut(id) {
            d.done = done;
            d.total = total;
        }
    }

    /// Resumable: bytes already in `<file>.part` are kept and the rest asked
    /// for with a Range request. Checked against the SHA-256 HuggingFace
    /// publishes for the file before it is renamed into place, so a truncated
    /// or corrupted download is never mistaken for a model.
    async fn fetch(&self, e: &Entry) -> Result<(), String> {
        tokio::fs::create_dir_all(&self.dir)
            .await
            .map_err(|err| format!("Cannot create {}: {err}", self.dir.display()))?;
        let (sha, size) = self.expected(e).await?;
        let part = self.path(e).with_extension("part");
        let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
        if have > size {
            have = 0;
        }
        let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{}", e.file);
        let mut req = self.http.get(&url);
        if have > 0 {
            req = req.header("Range", format!("bytes={have}-"));
        }
        let mut resp = req.send().await.map_err(|err| offline(&err))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(format!("The download server answered with an error ({status})."));
        }
        let resumed = status == reqwest::StatusCode::PARTIAL_CONTENT;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resumed)
            .truncate(!resumed)
            .open(&part)
            .await
            .map_err(|err| format!("Cannot write {}: {err}", part.display()))?;
        let mut done = if resumed { have } else { 0 };
        while let Some(chunk) = resp.chunk().await.map_err(|err| offline(&err))? {
            file.write_all(&chunk).await.map_err(|err| disk(&err))?;
            done += chunk.len() as u64;
            self.report(e.id, done, size);
        }
        file.flush().await.map_err(|err| disk(&err))?;
        drop(file);

        let part2 = part.clone();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&part2))
            .await
            .map_err(|err| err.to_string())??;
        if actual != sha {
            let _ = tokio::fs::remove_file(&part).await;
            return Err("The download was damaged (its checksum did not match). Please try again.".into());
        }
        tokio::fs::rename(&part, self.path(e)).await.map_err(|err| disk(&err))
    }

    /// The file's SHA-256 and size, from the HuggingFace tree listing.
    async fn expected(&self, e: &Entry) -> Result<(String, u64), String> {
        let url = format!("https://huggingface.co/api/models/{HF_REPO}/tree/main");
        let list: Value = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|err| offline(&err))?
            .json()
            .await
            .map_err(|err| format!("Unexpected answer from HuggingFace: {err}"))?;
        let item = list
            .as_array()
            .and_then(|a| a.iter().find(|i| i.get("path").and_then(Value::as_str) == Some(e.file)))
            .ok_or_else(|| format!("HuggingFace no longer lists {}.", e.file))?;
        let lfs = item.get("lfs").ok_or("HuggingFace gave no checksum for this file.")?;
        let sha = lfs.get("oid").and_then(Value::as_str).ok_or("HuggingFace gave no checksum for this file.")?;
        let size = lfs.get("size").and_then(Value::as_u64).unwrap_or(0);
        Ok((sha.to_owned(), size))
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn offline(e: &reqwest::Error) -> String {
    if e.is_connect() || e.is_timeout() {
        "Lana could not reach the download server. Check the internet connection and try again.".into()
    } else {
        format!("The download stopped: {e}")
    }
}

fn disk(e: &std::io::Error) -> String {
    if e.raw_os_error() == Some(28) || e.raw_os_error() == Some(112) {
        "The disk is full.".into()
    } else {
        format!("Could not save the model: {e}")
    }
}
