// SPDX-License-Identifier: GPL-3.0-or-later
// The catalogue of models Lana can download and run itself, and the downloads.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde_json::{json, Value};
use sha2::Sha256;
use tokio::io::AsyncWriteExt;

const WHISPER: &str = "ggerganov/whisper.cpp";
const SHERPA_SEG: &str = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0";
const SHERPA_EMB: &str = "csukuangfj/speaker-embedding-models";
const PARAKEET: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8";
const TAGGER: &str = "k2-fsa/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15";

/// Silero VAD v5 (MIT), as k2-fsa's maintainer publishes it for sherpa-onnx.
/// Not a catalogue entry: it is 2.3 MB, every recogniser cuts its pieces with
/// it, and it does nothing on its own, so asking the user to choose it would
/// only be one more thing to understand. It comes down with the first model
/// the user downloads (and with any later one, until it is here), from the
/// same host as the models, so the privacy claim is unchanged. Without it,
/// pieces are cut by energy as before (see vad.rs).
pub const VAD: File = File { repo: "csukuangfj/vad", path: "silero_vad_v5.onnx", local: "silero_vad_v5.onnx" };

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
    /// A NeMo transducer (Parakeet TDT) through sherpa-onnx: words from
    /// word pieces, with token times and durations.
    Transducer,
    /// An AudioSet tagger through sherpa-onnx: laughter, music, coughs.
    Tags,
}

impl Kind {
    /// The role the page fills a tier by, and the settings screen groups by.
    pub fn role(self) -> &'static str {
        match self {
            Kind::Whisper | Kind::Letters | Kind::Transducer => "words",
            Kind::Phones => "phones",
            Kind::Speakers => "speakers",
            Kind::Tags => "tags",
        }
    }
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
        Entry { id: "parakeet-tdt-0.6b-v3", kind: Kind::Transducer, size_mb: 646, dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: PARAKEET, path: "encoder.int8.onnx", local: "encoder.int8.onnx" },
                         File { repo: PARAKEET, path: "decoder.int8.onnx", local: "decoder.int8.onnx" },
                         File { repo: PARAKEET, path: "joiner.int8.onnx", local: "joiner.int8.onnx" },
                         File { repo: PARAKEET, path: "tokens.txt", local: "tokens.txt" }],
                label: "Parakeet v3",
                note: "Words with punctuation in 25 European languages (NVIDIA, 2025), which it detects itself. No Georgian or Caucasian languages." },
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
        Entry { id: "sounds", kind: Kind::Tags, size_mb: 27, dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: TAGGER, path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: TAGGER, path: "class_labels_indices.csv", local: "labels.csv" }],
                label: "Sounds",
                note: "Laughter, music, coughs, breathing and other sounds that are not words, from the 527 AudioSet classes: fills an events tier." },
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
    pub(crate) downloads: Arc<Mutex<HashMap<String, Download>>>,
    pub(crate) http: reqwest::Client,
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
                    "kind": e.kind.role(),
                    "licence": e.licence,
                    "installed": self.is_installed(e),
                    "downloading": d.filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
                    "error": d.and_then(|d| d.error.clone())
                        .or_else(|| (e.kind == Kind::Whisper).then(super::whisper_supported).and_then(Result::err)),
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
        tokio::spawn(async move {
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

    pub(crate) fn report(&self, id: &str, done: u64, total: u64) {
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
                let url = format!("https://huggingface.co/{}/resolve/main/{}", f.repo, f.path);
                self.fetch_file(e.id, &url, &dest, &sha, size, before, total).await?;
            }
            before += size;
        }
        if !self.vad_path().is_file() {
            if let Err(err) = self.fetch_vad().await {
                // The model itself is in place; pieces fall back to energy cuts.
                eprintln!("lana: VAD download failed: {err}");
            }
        }
        Ok(())
    }

    pub fn vad_path(&self) -> PathBuf {
        self.dir.join("vad").join(VAD.local)
    }

    /// The VAD model, verified like any other file. Its progress is reported
    /// under an id no download is listed by, so it does not move the bar of
    /// the model it came with.
    pub async fn fetch_vad(&self) -> Result<(), String> {
        let list = self.listing(VAD.repo).await?;
        let (sha, size) = expected(&list, &VAD)?;
        let url = format!("https://huggingface.co/{}/resolve/main/{}", VAD.repo, VAD.path);
        self.fetch_file("vad", &url, &self.vad_path(), &sha, size, 0, size).await
    }

    #[allow(clippy::too_many_arguments)]
    /// One file into place, resumable and verified. `sha` is a SHA-256, or
    /// `git:<sha1>` for a small file kept outside LFS, or empty to skip the
    /// check (only for a local test mirror; see mms.rs).
    pub(crate) async fn fetch_file(&self, id: &str, url: &str, dest: &Path, sha: &str, size: u64, before: u64, total: u64) -> Result<(), String> {
        let parent = dest.parent().unwrap_or(&self.dir);
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|err| format!("Cannot create {}: {err}", parent.display()))?;
        let part = part_of(dest);
        let mut have = tokio::fs::metadata(&part).await.map(|m| m.len()).unwrap_or(0);
        if size > 0 && have > size {
            have = 0;
        }
        let mut req = self.http.get(url);
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
        if !sha.is_empty() && !actual.split('|').any(|a| a == sha) {
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

/// The file's SHA-256 and its git blob id (SHA-1 over "blob <size>\0" and
/// the contents), as "<sha256>|git:<sha1>", in one pass. HuggingFace lists
/// an LFS file by the first and a small file kept in git by the second, and
/// "small" can still be over a megabyte (MMS's vocab.json is 1.3 MB).
fn sha256_file(path: &Path) -> Result<String, String> {
    use sha1::{Digest as _, Sha1};
    use std::io::Read;
    let len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let (mut h256, mut h1) = (Sha256::new(), Sha1::new());
    h1.update(format!("blob {len}\0").as_bytes());
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h256.update(&buf[..n]);
        h1.update(&buf[..n]);
    }
    let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
    Ok(format!("{}|git:{}", hex(&h256.finalize()), hex(&h1.finalize())))
}

pub(crate) fn offline(e: &reqwest::Error) -> String {
    if e.is_connect() || e.is_timeout() {
        "Lana could not reach the download server. Check the internet connection and try again.".into()
    } else {
        format!("The download stopped: {e}")
    }
}

pub(crate) fn disk(e: &std::io::Error) -> String {
    if e.raw_os_error() == Some(28) || e.raw_os_error() == Some(112) {
        "The disk is full.".into()
    } else {
        format!("Could not save the model: {e}")
    }
}
