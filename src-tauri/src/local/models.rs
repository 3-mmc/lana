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

const WHISPER: &str = "ggerganov/whisper.cpp";
const SHERPA_SEG: &str = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0";
const SHERPA_EMB: &str = "csukuangfj/speaker-embedding-models";

/// What a model does, which decides the engine that runs it and the tier it fills.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// whisper.cpp: words, with DTW word timings.
    Whisper,
    /// A CTC phone recogniser run by sherpa-onnx: IPA phones with frame times.
    Phones,
    /// Omnilingual ASR through sherpa-onnx: words from character CTC.
    Letters,
    /// pyannote segmentation + a speaker embedding, run by sherpa-onnx.
    Speakers,
}

/// One file of a model, from a HuggingFace repository.
pub struct File {
    pub repo: &'static str,
    pub path: &'static str,
    /// The name it is saved under; several models call their file model.onnx.
    pub local: &'static str,
}

pub struct Entry {
    pub id: &'static str,
    pub kind: Kind,
    pub files: &'static [File],
    pub label: &'static str,
    pub note: &'static str,
    pub licence: &'static str,
    pub size_mb: u32,
    /// whisper.cpp's alignment-head preset, which DTW word timing needs.
    pub dtw: whisper_rs::DtwModelPreset,
}

/// Short and opinionated. The first Whisper fits nearly every computer; the
/// last one is the best of these and still under 600 MB thanks to quantisation.
/// Every model here may be used for any purpose, commercial work included.
pub fn catalogue() -> Vec<Entry> {
    use whisper_rs::DtwModelPreset as P;
    vec![
        Entry { id: "whisper-base", kind: Kind::Whisper, size_mb: 142, dtw: P::Base, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-base.bin", local: "ggml-base.bin" }],
                label: "Whisper base",
                note: "Words. Small and quick. Good enough to try Lana out; misses more words in noisy or fast speech." },
        Entry { id: "whisper-small", kind: Kind::Whisper, size_mb: 466, dtw: P::Small, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-small.bin", local: "ggml-small.bin" }],
                label: "Whisper small",
                note: "Words. A good balance on an ordinary laptop." },
        Entry { id: "whisper-large-v3-turbo-q5", kind: Kind::Whisper, size_mb: 547, dtw: P::LargeV3Turbo, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-large-v3-turbo-q5_0.bin", local: "ggml-large-v3-turbo-q5_0.bin" }],
                label: "Whisper large-v3 turbo",
                note: "Words. The most accurate here, especially for languages other than English. Slower without a fast processor." },
        Entry { id: "omnilingual-300m", kind: Kind::Letters, size_mb: 365, dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", path: "tokens.txt", local: "tokens.txt" }],
                label: "Omnilingual 300M",
                note: "Words in some 1600 languages (Meta, 2025), as plain lowercase text. It is not told the language, and for languages it knows little it can write in the wrong alphabet: in testing, Dargwa came out in Arabic and Latin letters. Good where the alphabet is unambiguous, such as Georgian." },
        Entry { id: "omnilingual-1b", kind: Kind::Letters, size_mb: 1032, dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05", path: "tokens.txt", local: "tokens.txt" }],
                label: "Omnilingual 1B",
                note: "Words like Omnilingual 300M, with the same caveat about alphabets; more accurate on Georgian (2.2% character errors against 6.0% in testing). Needs 2 GB of memory while it runs." },
        Entry { id: "zipa-small", kind: Kind::Phones, size_mb: 71, dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: "anyspeech/zipa-small-crctc-500k", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "anyspeech/zipa-small-crctc-500k", path: "tokens.txt", local: "tokens.txt" }],
                label: "ZIPA small",
                note: "IPA phones read straight off the sound, in any language, with no dictionary. Small and fast." },
        Entry { id: "zipa-large", kind: Kind::Phones, size_mb: 310, dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: "anyspeech/zipa-large-crctc-500k", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "anyspeech/zipa-large-crctc-500k", path: "tokens.txt", local: "tokens.txt" }],
                label: "ZIPA large",
                note: "IPA phones, like ZIPA small. Larger; in testing it was not reliably better, so try small first." },
        Entry { id: "speakers", kind: Kind::Speakers, size_mb: 36, dtw: P::Base, licence: "MIT (segmentation), CC BY 4.0 (voices)",
                files: &[File { repo: SHERPA_SEG, path: "model.onnx", local: "segmentation.onnx" },
                         File { repo: SHERPA_EMB, path: "wespeaker_en_voxceleb_CAM++.onnx", local: "campplus.onnx" }],
                label: "Speakers",
                note: "Who is talking, by voice: fills the speaker tier, beside whichever recogniser runs." },
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

    /// Where one of a model's files lives. Whisper files sit directly in the
    /// models folder, where the first versions of Lana put them; everything
    /// else gets a folder of its own.
    pub fn file(&self, e: &Entry, f: &File) -> PathBuf {
        if e.kind == Kind::Whisper { self.dir.join(f.local) } else { self.dir.join(e.id).join(f.local) }
    }

    /// The model's only file, or its first: the one an engine is given.
    pub fn path(&self, e: &Entry) -> PathBuf {
        self.file(e, &e.files[0])
    }

    /// A file of the model by its saved name.
    pub fn named(&self, e: &Entry, local: &str) -> PathBuf {
        let f = e.files.iter().find(|f| f.local == local).expect("a file of this model");
        self.file(e, f)
    }

    pub fn is_installed(&self, e: &Entry) -> bool {
        e.files.iter().all(|f| self.file(e, f).is_file())
    }

    pub fn installed(&self) -> Vec<Entry> {
        catalogue().into_iter().filter(|e| self.is_installed(e)).collect()
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
                    "kind": match e.kind { Kind::Whisper | Kind::Letters => "words", Kind::Phones => "phones", Kind::Speakers => "speakers" },
                    "licence": e.licence,
                    "installed": self.is_installed(e),
                    "downloading": d.filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
                    "error": d.and_then(|d| d.error.clone()),
                })
            })
            .collect();
        json!({ "models": entries, "dir": self.dir.display().to_string() })
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let e = find(id).ok_or("no such model")?;
        for f in e.files {
            let p = self.file(&e, f);
            let _ = std::fs::remove_file(part_of(&p));
            match std::fs::remove_file(&p) {
                Ok(()) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                Err(err) => return Err(err.to_string()),
            }
        }
        if e.kind != Kind::Whisper {
            let _ = std::fs::remove_dir(self.dir.join(e.id));
        }
        Ok(())
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
    /// for with a Range request. Each file is checked against the SHA-256
    /// HuggingFace publishes for it before it is renamed into place, so a
    /// truncated or corrupted download is never mistaken for a model. Progress
    /// is reported over all of a model's files together.
    async fn fetch(&self, e: &Entry) -> Result<(), String> {
        let mut want = Vec::new();
        let mut lists: HashMap<&str, Value> = HashMap::new();
        for f in e.files {
            if !lists.contains_key(f.repo) {
                lists.insert(f.repo, self.listing(f.repo).await?);
            }
            want.push((f, expected(&lists[f.repo], f)?));
        }
        let total: u64 = want.iter().map(|(_, (_, size))| size).sum();
        let mut before = 0u64;
        for (f, (sha, size)) in want {
            let dest = self.file(e, f);
            if !dest.is_file() {
                self.fetch_file(e.id, f, &dest, &sha, size, before, total).await?;
            }
            before += size;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn fetch_file(&self, id: &str, f: &File, dest: &Path, sha: &str, size: u64, before: u64, total: u64) -> Result<(), String> {
        let parent = dest.parent().unwrap_or(&self.dir);
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| format!("Cannot create {}: {err}", parent.display()))?;
        let part = part_of(dest);
        let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
        if have > size {
            have = 0;
        }
        let url = format!("https://huggingface.co/{}/resolve/main/{}", f.repo, f.path);
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
            self.report(id, before + done, total);
        }
        file.flush().await.map_err(|err| disk(&err))?;
        drop(file);

        let part2 = part.clone();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&part2))
            .await
            .map_err(|err| err.to_string())??;
        if !actual.split('|').any(|a| a == sha) {
            let _ = tokio::fs::remove_file(&part).await;
            return Err("The download was damaged (its checksum did not match). Please try again.".into());
        }
        tokio::fs::rename(&part, dest).await.map_err(|err| disk(&err))
    }

    /// A repository's file listing, which carries each file's SHA-256 and size.
    async fn listing(&self, repo: &str) -> Result<Value, String> {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main");
        self.http
            .get(&url)
            .send()
            .await
            .map_err(|err| offline(&err))?
            .json()
            .await
            .map_err(|err| format!("Unexpected answer from HuggingFace: {err}"))
    }
}

/// `x.onnx` downloads into `x.onnx.part`.
fn part_of(p: &Path) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(".part");
    PathBuf::from(s)
}

/// The file's SHA-256 and size from a repository listing. Small files that
/// are not in LFS (tokens.txt) have only a git blob id, so for those the
/// checksum is the git one, recomputed the same way.
fn expected(list: &Value, f: &File) -> Result<(String, u64), String> {
    let item = list
        .as_array()
        .and_then(|a| a.iter().find(|i| i.get("path").and_then(Value::as_str) == Some(f.path)))
        .ok_or_else(|| format!("HuggingFace no longer lists {}.", f.path))?;
    if let Some(lfs) = item.get("lfs") {
        let sha = lfs.get("oid").and_then(Value::as_str).ok_or("HuggingFace gave no checksum for this file.")?;
        let size = lfs.get("size").and_then(Value::as_u64).unwrap_or(0);
        return Ok((sha.to_owned(), size));
    }
    let oid = item.get("oid").and_then(Value::as_str).ok_or("HuggingFace gave no checksum for this file.")?;
    let size = item.get("size").and_then(Value::as_u64).unwrap_or(0);
    Ok((format!("git:{oid}"), size))
}

/// SHA-256 of the file, or for a `git:` expectation the git blob id
/// (SHA-1 over "blob <size>\0" and the contents), which is what HuggingFace
/// lists for small files kept outside LFS.
fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes_len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    if bytes_len < 1 << 20 {
        // Small enough to hash both ways; the caller compares with whichever
        // it expects, so return the matching form below.
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let sha = Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect::<String>();
        return Ok(format!("{sha}|git:{}", git_blob_id(&data)));
    }
    sha256_large(path)
}

fn git_blob_id(data: &[u8]) -> String {
    use sha1::{Digest as _, Sha1};
    let mut h = Sha1::new();
    h.update(format!("blob {}\0", data.len()).as_bytes());
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_large(path: &Path) -> Result<String, String> {
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
