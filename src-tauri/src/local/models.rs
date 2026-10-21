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
/// The floating-point exports, for an accelerator. Their weights are in a
/// file beside the graph, which names it: it keeps its name here.
const PARAKEET_FP32: &str = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3";
const OMNI_300M_FP32: &str = "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12";
const OMNI_1B_FP32: &str = "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-2026-02-05";
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
    /// A CTC phone recogniser: IPA phones with frame times.
    Phones,
    /// Omnilingual ASR: words from character CTC.
    Letters,
    /// pyannote segmentation + a speaker embedding (pyannote.rs).
    Speakers,
    /// A NeMo transducer (Parakeet TDT): words from
    /// word pieces, with token times and durations.
    Transducer,
    /// An AudioSet tagger: laughter, music, coughs.
    Tags,
    /// CREPE: a second f0 track, drawn beside Praat's. It fills
    /// no tier, so the providers list leaves it out.
    Pitch,
    /// MMS-LID through ort: which language is spoken where (lid.rs).
    Lid,
}

impl Kind {
    /// The role the page fills a tier by, and the settings screen groups by.
    pub fn role(self) -> &'static str {
        match self {
            Kind::Whisper | Kind::Letters | Kind::Transducer => "words",
            Kind::Phones => "phones",
            Kind::Speakers => "speakers",
            Kind::Tags => "tags",
            Kind::Pitch => "pitch",
            Kind::Lid => "lid",
        }
    }

    /// Whether the model is under a non-commercial licence, which the user
    /// accepts once in the settings before it downloads (the same acceptance
    /// as MMS, from the same publisher under the same licence; see mms.rs).
    pub fn non_commercial(self) -> bool {
        self == Kind::Lid
    }
}

/// Files that are GitHub release assets rather than HuggingFace files: the
/// repo is written "github:<owner>/<repo>@<tag>". GitHub publishes no
/// checksum the app can check a download against, so each file's SHA-256 and
/// size are pinned here, measured from the release when it was added.
const PINNED: &[(&str, &str, &str, u64)] = &[
    (CREPE, "tiny.onnx", "91fc2a0fd10f965dbf7775995daf50e99273caedd7efd00001f23be649da1bc3", 1_955_762),
    (CREPE, "small.onnx", "cd119f6f5f608d9342a1c6f81e4653f21dbcde440a96d474ecbf6f27c6253466", 6_524_192),
];
const CREPE: &str = "github:yqzhishen/onnxcrepe@v1.1.0";

/// Where a file comes from, with its checksum and size when they are pinned.
fn source(f: &File) -> (String, Option<(String, u64)>) {
    match f.repo.strip_prefix("github:").and_then(|r| r.split_once('@')) {
        Some((repo, tag)) => {
            let pin = PINNED.iter().find(|p| p.0 == f.repo && p.1 == f.path).map(|p| (p.2.to_owned(), p.3));
            (format!("https://github.com/{repo}/releases/download/{tag}/{}", f.path), pin)
        }
        None => (format!("https://huggingface.co/{}/resolve/main/{}", f.repo, f.path), None),
    }
}

/// One file of a model, from a HuggingFace repository or a GitHub release.
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
    /// Versions in floating point for a GPU or the Neural Engine, which run
    /// int8 models no faster than the CPU (runtime.rs). An optional second
    /// download; the first file stands in for the model's first file.
    pub accel: &'static [Variant],
}

/// A model in another precision ("fp32", "fp16").
pub struct Variant {
    pub precision: &'static str,
    pub files: &'static [File],
    pub size_mb: u32,
}

/// Short and opinionated. The first Whisper fits nearly every computer; the
/// last one is the best of these and still under 600 MB thanks to quantisation.
/// Every model here may be used for any purpose, commercial work included,
/// except MMS-LID (non-commercial, behind the same acceptance as MMS).
pub fn catalogue() -> Vec<Entry> {
    use whisper_rs::DtwModelPreset as P;
    vec![
        Entry { id: "whisper-base", kind: Kind::Whisper, size_mb: 142, accel: &[], dtw: P::Base, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-base.bin", local: "ggml-base.bin" }],
                label: "Whisper base",
                note: "Words. Small and quick. Good enough to try Lana out; misses more words in noisy or fast speech." },
        Entry { id: "whisper-small", kind: Kind::Whisper, size_mb: 466, accel: &[], dtw: P::Small, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-small.bin", local: "ggml-small.bin" }],
                label: "Whisper small",
                note: "Words. A good balance on an ordinary laptop." },
        Entry { id: "whisper-large-v3-turbo-q5", kind: Kind::Whisper, size_mb: 547, accel: &[], dtw: P::LargeV3Turbo, licence: "MIT",
                files: &[File { repo: WHISPER, path: "ggml-large-v3-turbo-q5_0.bin", local: "ggml-large-v3-turbo-q5_0.bin" }],
                label: "Whisper large-v3 turbo",
                note: "Words. The most accurate here, especially for languages other than English. Slower without a fast processor." },
        Entry { id: "omnilingual-300m", kind: Kind::Letters, size_mb: 365, accel: &[Variant { precision: "fp32", size_mb: 1303, files: &[File { repo: OMNI_300M_FP32, path: "model.onnx", local: "model.fp32.onnx" }] }], dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12", path: "tokens.txt", local: "tokens.txt" }],
                label: "Omnilingual 300M",
                note: "Words in some 1600 languages (Meta, 2025), as plain lowercase text. Recognition can use the wrong alphabet for languages it knows less well." },
        Entry { id: "omnilingual-1b", kind: Kind::Letters, size_mb: 1032, accel: &[Variant { precision: "fp32", size_mb: 3904, files: &[File { repo: OMNI_1B_FP32, path: "model.onnx", local: "model.fp32.onnx" },
                                                                         File { repo: OMNI_1B_FP32, path: "model.weights", local: "model.weights" }] }], dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "csukuangfj2/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-v2-int8-2026-02-05", path: "tokens.txt", local: "tokens.txt" }],
                label: "Omnilingual 1B",
                note: "Words like Omnilingual 300M, with a larger model and the same caveat about alphabets. Needs 2 GB of memory while it runs." },
        Entry { id: "parakeet-tdt-0.6b-v3", kind: Kind::Transducer, size_mb: 646, accel: &[Variant { precision: "fp32", size_mb: 2477, files: &[File { repo: PARAKEET_FP32, path: "encoder.onnx", local: "encoder.fp32.onnx" },
                                                                         File { repo: PARAKEET_FP32, path: "encoder.weights", local: "encoder.weights" }] }], dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: PARAKEET, path: "encoder.int8.onnx", local: "encoder.int8.onnx" },
                         File { repo: PARAKEET, path: "decoder.int8.onnx", local: "decoder.int8.onnx" },
                         File { repo: PARAKEET, path: "joiner.int8.onnx", local: "joiner.int8.onnx" },
                         File { repo: PARAKEET, path: "tokens.txt", local: "tokens.txt" }],
                label: "Parakeet v3",
                note: "Words with punctuation in 25 European languages (NVIDIA, 2025), which it detects itself." },
        Entry { id: "zipa-small", kind: Kind::Phones, size_mb: 71, accel: &[], dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: "anyspeech/zipa-small-crctc-500k", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "anyspeech/zipa-small-crctc-500k", path: "tokens.txt", local: "tokens.txt" }],
                label: "ZIPA small",
                note: "IPA phones read straight off the sound, in any language, with no dictionary. Small and fast." },
        Entry { id: "zipa-large", kind: Kind::Phones, size_mb: 310, accel: &[Variant { precision: "fp32", size_mb: 1205, files: &[File { repo: "anyspeech/zipa-large-crctc-500k", path: "model.onnx", local: "model.fp32.onnx" }] },
                        Variant { precision: "fp16", size_mb: 604, files: &[File { repo: "anyspeech/zipa-large-crctc-500k", path: "model.fp16.onnx", local: "model.fp16.onnx" }] }], dtw: P::Base, licence: "CC BY 4.0",
                files: &[File { repo: "anyspeech/zipa-large-crctc-500k", path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: "anyspeech/zipa-large-crctc-500k", path: "tokens.txt", local: "tokens.txt" }],
                label: "ZIPA large",
                note: "IPA phones, like ZIPA small. Larger; in testing it was not reliably better, so try small first." },
        Entry { id: "speakers", kind: Kind::Speakers, size_mb: 36, accel: &[], dtw: P::Base, licence: "MIT (segmentation), CC BY 4.0 (voices)",
                files: &[File { repo: SHERPA_SEG, path: "model.onnx", local: "segmentation.onnx" },
                         File { repo: SHERPA_EMB, path: "wespeaker_en_voxceleb_CAM++.onnx", local: "campplus.onnx" }],
                label: "Speakers",
                note: "Who is talking, by voice: fills the speaker tier, beside whichever recogniser runs." },
        Entry { id: "sounds", kind: Kind::Tags, size_mb: 27, accel: &[], dtw: P::Base, licence: "Apache 2.0",
                files: &[File { repo: TAGGER, path: "model.int8.onnx", local: "model.int8.onnx" },
                         File { repo: TAGGER, path: "class_labels_indices.csv", local: "labels.csv" }],
                label: "Sounds",
                note: "Laughter, music, coughs, breathing and other sounds that are not words, from the 527 AudioSet classes: fills an events tier." },
        Entry { id: "crepe-tiny", kind: Kind::Pitch, size_mb: 2, accel: &[], dtw: P::Base, licence: "MIT",
                files: &[File { repo: CREPE, path: "tiny.onnx", local: "model.onnx" }],
                label: "CREPE tiny",
                note: "A second f0 track, read from the waveform by a network rather than by autocorrelation, drawn beside Praat's: where the two part, in creak or noise, look closer. Agreed with Praat's median f0 on both demo clips; about 5x real time." },
        Entry { id: "crepe-small", kind: Kind::Pitch, size_mb: 7, accel: &[], dtw: P::Base, licence: "MIT",
                files: &[File { repo: CREPE, path: "small.onnx", local: "model.onnx" }],
                label: "CREPE small",
                note: "CREPE, larger and slower (about 2.5x real time); try tiny first." },
        Entry { id: "langid-mms-1024", kind: Kind::Lid, size_mb: 975, accel: &[], dtw: P::Base, licence: super::mms::LICENCE,
                files: &[File { repo: "Xenova/mms-lid-1024", path: "onnx/model_quantized.onnx", local: "model.int8.onnx" },
                         File { repo: "Xenova/mms-lid-1024", path: "config.json", local: "config.json" }],
                label: "MMS-LID 1024",
                note: "Spoken language identification in 1,024 languages (Meta, 2023): fills the language tier of a recording marked as codeswitching, a few seconds at a time, so each stretch can be read in its own language. Non-commercial." },
    ]
}

/// A model's accelerator variant downloads and deletes under its id with
/// this added: "zipa-small+gpu".
pub const ACCEL_SUFFIX: &str = "+gpu";

pub fn accel_id(id: &str) -> String {
    format!("{id}{ACCEL_SUFFIX}")
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

    /// The variant this computer's accelerator would use, by its preference
    /// (runtime::float_preference), installed or not.
    pub fn variant<'a>(&self, e: &'a Entry) -> Option<&'a Variant> {
        if cfg!(target_os = "macos") {
            return super::mac_files::variant(e.id).filter(|_| super::mac_models::available(e.id));
        }
        super::runtime::float_preference().iter().find_map(|p| e.accel.iter().find(|v| v.precision == *p))
    }

    fn variant_installed(&self, e: &Entry, v: &Variant) -> bool {
        v.files.iter().all(|f| self.file(e, f).is_file())
    }

    /// The file a model's engine is given: its accelerator variant when the
    /// accelerator is in use and the variant is here, else its first file.
    pub fn model_file(&self, e: &Entry) -> PathBuf {
        if super::runtime::wants_float() || (cfg!(target_os = "macos") && super::runtime::compute() != super::runtime::Compute::Cpu) {
            if let Some(v) = self.variant(e).filter(|v| self.variant_installed(e, v)) {
                return self.file(e, &v.files[0]);
            }
        }
        self.path(e)
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
        let accelerated = super::runtime::has_accelerator();
        let entries: Vec<Value> = catalogue()
            .iter()
            .map(|e| {
                let d = downloads.get(e.id);
                json!({
                    "id": e.id, "label": e.label, "note": e.note, "size_mb": e.size_mb,
                    "kind": e.kind.role(),
                    "base_label": if e.kind == Kind::Whisper { "Shared CPU / GPU model" } else { "CPU model" },
                    "base_note": if e.kind == Kind::Whisper {
                        "One download for CPU and Automatic modes; no extra acceleration download."
                    } else if self.variant(e).is_some() {
                        "Required for CPU mode, correction and fallback. Remove all also removes acceleration downloads."
                    } else { "Used in CPU and Automatic modes." },
                    "licence": e.licence,
                    "nc": e.kind.non_commercial(),
                    "accepted": !e.kind.non_commercial() || self.mms_accepted(),
                    "installed": self.is_installed(e),
                    "downloading": d.filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
                    "error": d.and_then(|d| d.error.clone())
                        .or_else(|| (e.kind == Kind::Whisper).then(super::whisper_supported).and_then(Result::err)),
                    "accel": self.variant(e).filter(|_| accelerated).map(|v| {
                        let d = downloads.get(&accel_id(e.id));
                        json!({ "precision": v.precision, "size_mb": v.size_mb, "installed": self.variant_installed(e, v),
                                "label": if cfg!(target_os = "macos") { format!("Mac acceleration ({})", v.precision) } else { format!("GPU version ({})", v.precision) },
                                "downloading": d.filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
                                "error": d.and_then(|d| d.error.clone()) })
                    }),
                    "accel_note": if cfg!(target_os = "macos") && e.id.starts_with("omnilingual") && !super::mac_models::available(e.id) {
                        Some("Runs on the CPU. MLX acceleration requires macOS 15 or later and the bundled Mac runner.")
                    } else { None },
                })
            })
            .collect();
        json!({ "models": entries, "dir": self.dir.display().to_string() })
    }

    pub fn delete(&self, id: &str) -> Result<(), String> {
        let (base, accel) = match id.strip_suffix(ACCEL_SUFFIX) {
            Some(base) => (base, true),
            None => (id, false),
        };
        let e = find(base).ok_or("no such model")?;
        let mut downloads = self.downloads.lock().unwrap();
        let accelerator = accel_id(base);
        if [id, accelerator.as_str()].iter().any(|key| {
            downloads.get(*key).is_some_and(|d| d.error.is_none())
        }) {
            return Err("This model is still downloading. Wait for it to finish before removing it.".into());
        }
        let files = e.accel.iter().chain(super::mac_files::variant(e.id))
            .flat_map(|v| v.files.iter())
            .chain(if accel { &[][..] } else { e.files });
        for f in files {
            let p = self.file(&e, f);
            for path in [part_of(&p), p] {
                match std::fs::remove_file(&path) {
                    Ok(()) => {}
                    Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
                    Err(err) => return Err(format!("Could not remove {}: {err}", path.display())),
                }
            }
        }
        downloads.remove(id);
        downloads.remove(&accelerator);
        if !accel && e.kind != Kind::Whisper {
            let _ = std::fs::remove_dir(self.dir.join(e.id));
        }
        Ok(())
    }

    /// Start a download in the background. Returns at once; poll `list()`.
    pub fn start(&self, id: &str) -> Result<(), String> {
        let (base, accel) = match id.strip_suffix(ACCEL_SUFFIX) { Some(b) => (b, true), None => (id, false) };
        let e = find(base).ok_or("no such model")?;
        let files: &'static [File] = if accel {
            self.variant(&e).ok_or("This computer has no accelerator that version is for.")?.files
        } else {
            e.files
        };
        if e.kind.non_commercial() && !self.mms_accepted() {
            return Err(format!("{} is for non-commercial use only: accept its licence first.", e.label));
        }
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
            let result = me.fetch(&e, &id, files).await;
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
    async fn fetch(&self, e: &Entry, id: &str, files: &'static [File]) -> Result<(), String> {
        let mut want = Vec::new();
        let mut lists: HashMap<&str, Value> = HashMap::new();
        for f in files {
            if f.repo.starts_with("github:") {
                let pin = source(f).1.ok_or_else(|| format!("{} has no pinned checksum.", f.path))?;
                want.push((f, pin));
                continue;
            }
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
                self.fetch_file(id, &source(f).0, &dest, &sha, size, before, total).await?;
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
    /// Include subfolders: MMS-LID keeps its model in `onnx/`.
    async fn listing(&self, repo: &str) -> Result<Value, String> {
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
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

#[cfg(test)]
mod removal_tests {
    use super::*;

    #[test]
    fn removal_restores_download_state_and_keeps_the_base_for_accel_only() {
        let dir = std::env::temp_dir().join(format!(
            "lana-removal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        for model_id in ["zipa-large", "omnilingual-300m", "parakeet-tdt-0.6b-v3"] {
            let models = Models::new(dir.join(model_id));
            let e = find(model_id).unwrap();
            let acceleration = super::super::mac_files::variant(e.id).unwrap_or(&e.accel[0]);
            for f in e.files.iter().chain(acceleration.files) {
                let p = models.file(&e, f);
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(p, b"test model").unwrap();
            }
            let id = accel_id(e.id);
            models.downloads.lock().unwrap().insert(
                id.clone(),
                Download {
                    error: Some("old error".into()),
                    ..Default::default()
                },
            );
            models.delete(&id).unwrap();
            assert!(models.is_installed(&e));
            assert!(!models.variant_installed(&e, acceleration));
            assert!(!models.downloads.lock().unwrap().contains_key(&id));
            models
                .downloads
                .lock()
                .unwrap()
                .insert(e.id.into(), Download::default());
            assert!(models
                .delete(e.id)
                .unwrap_err()
                .contains("still downloading"));
            assert!(models.is_installed(&e));
            models.downloads.lock().unwrap().clear();
            models.delete(e.id).unwrap();
            let listed = models.list();
            let entry = listed["models"]
                .as_array()
                .unwrap()
                .iter()
                .find(|m| m["id"] == e.id)
                .unwrap();
            assert_eq!(entry["installed"], false);
            assert!(entry["downloading"].is_null());
            assert!(entry["error"].is_null());
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn acceleration_removal_reports_file_errors() {
        let dir = std::env::temp_dir().join(format!(
            "lana-removal-error-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let models = Models::new(dir.clone());
        let e = find("zipa-large").unwrap();
        // A directory where a model file should be makes unlink fail on every OS.
        std::fs::create_dir_all(models.file(&e, &e.accel[0].files[0])).unwrap();
        assert!(models
            .delete(&accel_id(e.id))
            .unwrap_err()
            .contains("Could not remove"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
