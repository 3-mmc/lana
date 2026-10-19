// SPDX-License-Identifier: GPL-3.0-or-later
// MMS (Meta, 2023): words in 1,159 languages (1,198 adapters, some per script
// or dialect), one small adapter each.
//
// MMS-1b-all is one large shared network plus, per language, a few MB of
// adapter layers and an output layer for that language's alphabet. Lana keeps
// that shape on disk:
//
//   mms/model.onnx, mms/mms-base.bin   the shared part, converted once by
//                                      Lana (MatMuls int8) and downloaded once
//   mms/adapters.json                  where each adapter tensor goes
//   mms/vocab.json                     every language's alphabet (Meta's)
//   mms/<lang>/adapter.bin             built here from Meta's own
//                                      adapter.<lang>.safetensors
//   mms/<lang>/tokens.txt              that language's alphabet
//
// ONNX Runtime opens mms/<lang>/model.onnx, which names its weights by file:
// mms-base.bin and adapter.bin, beside it. So each language folder holds hard
// links to the shared two (no extra space) and its own adapter.bin.
//
// MMS is CC BY-NC 4.0. Nothing here downloads until the user has accepted
// that licence in the settings, and every model name says "non-commercial".

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use serde_json::{json, Value};

use super::models::{disk, offline, Download, Models};

pub const LICENCE: &str = "CC BY-NC 4.0";
pub const LICENCE_URL: &str = "https://creativecommons.org/licenses/by-nc/4.0/";
const META: &str = "https://huggingface.co/facebook/mms-1b-all/resolve/main";
/// Lana's conversion of the shared part. LANA_MMS_BASE points elsewhere, such
/// as a local mirror for testing; downloads from an override are not
/// checksummed unless the server sends checksums.
const SHARED: &str = "https://huggingface.co/3mmc/lana-mms-1b-all-onnx/resolve/main";
const SHARED_FILES: &[&str] = &["model.onnx", "mms-base.bin", "adapters.json"];
/// The graph with a 4,608-symbol output layer, and its adapter layout, for the
/// four alphabets the 512-symbol graph cannot hold: Mandarin (4,495 symbols),
/// Cantonese, Japanese and Korean. Same weights in mms-base.bin; fetched only
/// when one of those languages is added (lana-models mms/widen.py makes them).
const WIDE_FILES: &[&str] = &["model-wide.onnx", "adapters-wide.json"];
pub const SHARED_MB: u32 = 1018;
const ADAPTER_MB: u32 = 9;

/// (code, English name), from MMS's vocab.json and the ISO 639-3 tables.
pub fn languages() -> &'static Vec<(String, String)> {
    static L: OnceLock<Vec<(String, String)>> = OnceLock::new();
    L.get_or_init(|| serde_json::from_str(include_str!("mms_languages.json")).expect("bundled language list"))
}

pub fn name_of(code: &str) -> Option<&'static str> {
    languages().iter().find(|(c, _)| c == code).map(|(_, n)| n.as_str())
}

fn shared_base() -> String {
    std::env::var("LANA_MMS_BASE").unwrap_or_else(|_| SHARED.into())
}

impl Models {
    pub fn mms_dir(&self) -> PathBuf {
        self.dir.join("mms")
    }

    fn accepted_file(&self) -> PathBuf {
        self.mms_dir().join("licence-accepted.json")
    }

    pub fn mms_accepted(&self) -> bool {
        self.accepted_file().is_file()
    }

    pub fn mms_accept(&self) -> Result<(), String> {
        std::fs::create_dir_all(self.mms_dir()).map_err(|e| disk(&e))?;
        let note = json!({ "licence": LICENCE, "url": LICENCE_URL, "accepted": chrono::Local::now().to_rfc3339() });
        std::fs::write(self.accepted_file(), serde_json::to_string_pretty(&note).unwrap()).map_err(|e| disk(&e))
    }

    fn shared_installed(&self) -> bool {
        SHARED_FILES.iter().chain(["vocab.json"].iter()).all(|f| self.mms_dir().join(f).is_file())
    }

    pub fn mms_lang_dir(&self, code: &str) -> PathBuf {
        self.mms_dir().join(code)
    }

    pub fn mms_lang_installed(&self, code: &str) -> bool {
        let d = self.mms_lang_dir(code);
        ["model.onnx", "mms-base.bin", "adapter.bin", "tokens.txt"].iter().all(|f| d.join(f).is_file())
    }

    pub fn mms_installed(&self) -> Vec<String> {
        let mut out: Vec<String> = std::fs::read_dir(self.mms_dir())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|c| name_of(c).is_some() && self.mms_lang_installed(c))
            .collect();
        out.sort();
        out
    }

    /// Everything the settings screen shows about MMS.
    pub fn mms_status(&self) -> Value {
        let dl = self.downloads.lock().unwrap().clone();
        let progress = |k: &str| {
            dl.get(k).filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total }))
        };
        let downloading: HashMap<String, Value> = dl
            .iter()
            .filter(|(k, d)| k.starts_with("mms:") && *k != "mms:shared" && d.error.is_none())
            .map(|(k, d)| (k[4..].to_owned(), json!({ "done": d.done, "total": d.total })))
            .collect();
        let errors: HashMap<String, String> = dl
            .iter()
            .filter(|(k, _)| k.starts_with("mms:"))
            .filter_map(|(k, d)| d.error.clone().map(|e| (k[4..].to_owned(), e)))
            .collect();
        json!({
            "licence": LICENCE, "licence_url": LICENCE_URL, "accepted": self.mms_accepted(),
            "shared": { "installed": self.shared_installed(), "size_mb": SHARED_MB, "downloading": progress("mms:shared") },
            "adapter_mb": ADAPTER_MB,
            "installed": self.mms_installed().iter().map(|c| json!({ "code": c, "name": name_of(c) })).collect::<Vec<_>>(),
            "downloading": downloading,
            "errors": errors,
        })
    }

    /// Download a language, and the shared part first if it is missing.
    pub fn mms_start(&self, code: &str) -> Result<(), String> {
        if !self.mms_accepted() {
            return Err("Accept the MMS licence first.".into());
        }
        if name_of(code).is_none() {
            return Err(format!("MMS has no language '{code}'."));
        }
        let key = format!("mms:{code}");
        {
            let mut dl = self.downloads.lock().unwrap();
            if dl.get(&key).is_some_and(|d| d.error.is_none()) {
                return Ok(());
            }
            dl.insert(key.clone(), Download::default());
        }
        let me = self.clone();
        let code = code.to_owned();
        tokio::spawn(async move {
            let result = me.mms_fetch(&code).await;
            let mut dl = me.downloads.lock().unwrap();
            match result {
                Ok(()) => {
                    dl.remove(&format!("mms:{code}"));
                }
                Err(msg) => {
                    dl.insert(format!("mms:{code}"), Download { error: Some(msg), ..Default::default() });
                }
            }
        });
        Ok(())
    }

    async fn mms_fetch(&self, code: &str) -> Result<(), String> {
        // One shared download at a time: three languages added in a row used
        // to fetch it three times over, into the same .part files.
        static SHARED_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        {
            let _one = SHARED_LOCK.lock().await;
            if !self.shared_installed() {
                self.mms_fetch_shared().await?;
            }
        }
        // An alphabet larger than the shared graph's output layer needs the wide graph.
        let symbols = {
            let vocab: Value = serde_json::from_slice(&std::fs::read(self.mms_dir().join("vocab.json")).map_err(|e| disk(&e))?)
                .map_err(|e| format!("vocab.json: {e}"))?;
            vocab.get(code).and_then(Value::as_object).map_or(0, |v| v.len())
        };
        let layout_head = {
            let layout: Value = serde_json::from_slice(&std::fs::read(self.mms_dir().join("adapters.json")).map_err(|e| disk(&e))?)
                .map_err(|e| format!("adapters.json: {e}"))?;
            layout["head"].as_u64().unwrap_or(512) as usize
        };
        let wide = symbols > layout_head;
        if wide {
            let _one = SHARED_LOCK.lock().await;
            for f in WIDE_FILES {
                let dest = self.mms_dir().join(f);
                if !dest.is_file() {
                    let url = format!("{}/{f}", shared_base());
                    let (sha, size) = self.head(&url).await?;
                    self.fetch_file(&format!("mms:{code}"), &url, &dest, &sha, size, 0, size).await?;
                }
            }
        }
        let dir = self.mms_lang_dir(code);
        tokio::fs::create_dir_all(&dir).await.map_err(|e| disk(&e))?;
        let url = format!("{META}/adapter.{code}.safetensors");
        let (sha, size) = self.head(&url).await?;
        let st = dir.join("adapter.safetensors");
        self.fetch_file(&format!("mms:{code}"), &url, &st, &sha, size, 0, size).await?;

        // Build adapter.bin and tokens.txt, then link the shared part in.
        let mms = self.mms_dir();
        let code2 = code.to_owned();
        tokio::task::spawn_blocking(move || -> Result<(), String> {
            let (graph, layout_file) = if wide { ("model-wide.onnx", "adapters-wide.json") } else { ("model.onnx", "adapters.json") };
            let layout: Value = serde_json::from_slice(&std::fs::read(mms.join(layout_file)).map_err(|e| disk(&e))?)
                .map_err(|e| format!("{layout_file}: {e}"))?;
            let adapter = build_adapter(&std::fs::read(&st).map_err(|e| disk(&e))?, &layout)?;
            let vocab: Value = serde_json::from_slice(&std::fs::read(mms.join("vocab.json")).map_err(|e| disk(&e))?)
                .map_err(|e| format!("vocab.json: {e}"))?;
            let tokens = tokens_for(&vocab, &code2)?;
            let dir = mms.join(&code2);
            std::fs::write(dir.join("adapter.bin"), adapter).map_err(|e| disk(&e))?;
            std::fs::write(dir.join("tokens.txt"), tokens).map_err(|e| disk(&e))?;
            link_or_copy(&mms.join(graph), &dir.join("model.onnx"))?;
            link_or_copy(&mms.join("mms-base.bin"), &dir.join("mms-base.bin"))?;
            let _ = std::fs::remove_file(&st);
            Ok(())
        })
        .await
        .map_err(|e| e.to_string())?
    }

    /// The shared part: Lana's conversion, plus Meta's vocab.json.
    async fn mms_fetch_shared(&self) -> Result<(), String> {
        let key = "mms:shared";
        self.downloads.lock().unwrap().insert(key.into(), Download::default());
        let result = async {
            let base = shared_base();
            let mut want = Vec::new();
            for f in SHARED_FILES {
                let url = format!("{base}/{f}");
                want.push((url.clone(), *f, self.head(&url).await?));
            }
            let url = format!("{META}/vocab.json");
            want.push((url.clone(), "vocab.json", self.head(&url).await?));
            let total: u64 = want.iter().map(|w| w.2 .1).sum();
            let mut before = 0;
            for (url, f, (sha, size)) in want {
                let dest = self.mms_dir().join(f);
                if !dest.is_file() {
                    self.fetch_file(key, &url, &dest, &sha, size, before, total).await?;
                }
                before += size;
            }
            Ok::<(), String>(())
        }
        .await;
        let mut dl = self.downloads.lock().unwrap();
        match &result {
            Ok(()) => {
                dl.remove(key);
            }
            Err(e) => {
                dl.insert(key.into(), Download { error: Some(e.clone()), ..Default::default() });
            }
        }
        result
    }

    /// A file's checksum and size from the headers HuggingFace sends: the
    /// SHA-256 of an LFS file, or the git blob id of a small one.
    async fn head(&self, url: &str) -> Result<(String, u64), String> {
        // Not following the redirect: HuggingFace puts the checksum on the
        // redirect itself, and the storage it points to does not repeat it.
        static NOFOLLOW: OnceLock<reqwest::Client> = OnceLock::new();
        let client = NOFOLLOW.get_or_init(|| {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .user_agent(concat!("lana-desktop/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("HTTP client")
        });
        let r = client.head(url).send().await.map_err(|e| offline(&e))?;
        if r.status() == reqwest::StatusCode::NOT_FOUND {
            return Err("The download server does not have this file.".into());
        }
        let h = |k: &str| r.headers().get(k).and_then(|v| v.to_str().ok()).map(|v| v.trim_matches('"').to_owned());
        let size = h("x-linked-size").or_else(|| h("content-length")).and_then(|v| v.parse().ok()).unwrap_or(0);
        let sha = match h("x-linked-etag").or_else(|| h("x-repo-commit").and(h("etag"))) {
            Some(e) if e.len() == 64 => e,
            Some(e) if e.len() == 40 => format!("git:{e}"),
            _ if std::env::var_os("LANA_MMS_BASE").is_some() => String::new(),
            _ => return Err("The download server gave no checksum for this file.".into()),
        };
        Ok((sha, size))
    }

    pub fn mms_delete(&self, code: &str) -> Result<(), String> {
        if name_of(code).is_none() {
            return Err("no such language".into());
        }
        match std::fs::remove_dir_all(self.mms_lang_dir(code)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        }
    }

    /// Everything MMS, shared part included. The licence acceptance stays.
    pub fn mms_delete_all(&self) -> Result<(), String> {
        for e in std::fs::read_dir(self.mms_dir()).into_iter().flatten().flatten() {
            let p = e.path();
            if p == self.accepted_file() {
                continue;
            }
            let r = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
            r.map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// A hard link where the file system has them (NTFS, APFS, ext4), so a
/// language costs its adapter and not another gigabyte; a copy otherwise.
fn link_or_copy(from: &Path, to: &Path) -> Result<(), String> {
    let _ = std::fs::remove_file(to);
    if std::fs::hard_link(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map(|_| ()).map_err(|e| disk(&e))
}

/// tokens.txt in sherpa-onnx's form ("symbol id"), from MMS's vocab.json.
fn tokens_for(vocab: &Value, code: &str) -> Result<String, String> {
    let v = vocab.get(code).and_then(Value::as_object).ok_or_else(|| format!("vocab.json has no '{code}'"))?;
    let mut items: Vec<(&String, u64)> = v.iter().filter_map(|(s, i)| i.as_u64().map(|i| (s, i))).collect();
    items.sort_by_key(|(_, i)| *i);
    Ok(items.iter().map(|(s, i)| format!("{} {i}\n", if s.as_str() == " " { "|" } else { s.as_str() })).collect())
}

/// Meta's adapter.<lang>.safetensors rearranged into adapter.bin as
/// adapters.json describes: each tensor at its offset, transposed where the
/// export turned a Linear into a MatMul, and the output layer padded to the
/// fixed alphabet size the shared graph expects.
fn build_adapter(st: &[u8], layout: &Value) -> Result<Vec<u8>, String> {
    let tensors = parse_safetensors(st)?;
    let head = layout["head"].as_u64().ok_or("adapters.json: no head")? as usize;
    let mut out = vec![0u8; layout["adapter_bytes"].as_u64().ok_or("adapters.json: no size")? as usize];
    for t in layout["tensors"].as_array().ok_or("adapters.json: no tensors")? {
        let src = t["source"].as_str().unwrap_or("");
        let (shape, data) = tensors.get(src).ok_or_else(|| format!("the adapter has no tensor {src}"))?;
        let mut shape = shape.clone();
        let mut v = data.clone();
        if t["transpose"].as_bool() == Some(true) && shape.len() == 2 {
            let (r, c) = (shape[0], shape[1]);
            let mut tr = vec![0f32; v.len()];
            for i in 0..r {
                for j in 0..c {
                    tr[j * r + i] = v[i * c + j];
                }
            }
            v = tr;
            shape = vec![c, r];
        }
        match t["pad"].as_str() {
            Some("cols0") if shape.len() == 2 && shape[1] <= head => {
                let (r, c) = (shape[0], shape[1]);
                let mut p = vec![0f32; r * head];
                for i in 0..r {
                    p[i * head..i * head + c].copy_from_slice(&v[i * c..i * c + c]);
                }
                v = p;
                shape = vec![r, head];
            }
            Some("fill-1e4") if shape.len() == 1 && shape[0] <= head => {
                v.resize(head, -1e4);
                shape = vec![head];
            }
            None => {}
            Some(p) => return Err(format!("cannot pad {src} ({p}) to {head}")),
        }
        let want: Vec<usize> = t["shape"].as_array().unwrap_or(&vec![]).iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        if shape != want {
            return Err(format!("{src} has shape {shape:?}, expected {want:?}"));
        }
        let off = t["offset"].as_u64().unwrap_or(0) as usize;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        out.get_mut(off..off + bytes.len()).ok_or("adapters.json: offset past the end")?.copy_from_slice(&bytes);
    }
    Ok(out)
}

/// The safetensors format: an 8-byte little-endian header length, a JSON
/// header of name -> {dtype, shape, data_offsets}, then the raw data.
fn parse_safetensors(b: &[u8]) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>, String> {
    let bad = || "The adapter file is damaged.".to_owned();
    let n = u64::from_le_bytes(b.get(..8).ok_or_else(bad)?.try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(b.get(8..8 + n).ok_or_else(bad)?).map_err(|_| bad())?;
    let data = &b[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in header.as_object().ok_or_else(bad)? {
        if k == "__metadata__" {
            continue;
        }
        if v["dtype"] != "F32" {
            return Err(format!("unexpected dtype {} for {k}", v["dtype"]));
        }
        let shape: Vec<usize> = v["shape"].as_array().ok_or_else(bad)?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        let off: Vec<usize> = v["data_offsets"].as_array().ok_or_else(bad)?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect();
        let raw = data.get(off[0]..off[1]).ok_or_else(bad)?;
        let vals: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        out.insert(k.clone(), (shape, vals));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (k, shape, v) in tensors {
            let a = data.len();
            data.extend(v.iter().flat_map(|x| x.to_le_bytes()));
            header.insert(k.to_string(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [a, data.len()] }));
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend(h);
        out.extend(data);
        out
    }

    #[test]
    fn adapters_are_transposed_and_the_head_padded() {
        // A 2x3 weight exported transposed (3x2), and a head of 2 symbols padded to 4.
        let st = safetensors(&[
            ("w", vec![2, 3], vec![1., 2., 3., 4., 5., 6.]),
            ("lm_head.weight", vec![2, 3], vec![1., 2., 3., 4., 5., 6.]),
            ("lm_head.bias", vec![2], vec![0.5, 0.25]),
        ]);
        let layout = json!({ "head": 4, "adapter_bytes": 24 + 48 + 16, "tensors": [
            { "source": "w", "transpose": true, "pad": null, "shape": [3, 2], "offset": 0 },
            { "source": "lm_head.weight", "transpose": true, "pad": "cols0", "shape": [3, 4], "offset": 24 },
            { "source": "lm_head.bias", "transpose": false, "pad": "fill-1e4", "shape": [4], "offset": 72 },
        ]});
        let out = build_adapter(&st, &layout).unwrap();
        let f: Vec<f32> = out.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(&f[..6], &[1., 4., 2., 5., 3., 6.]);
        assert_eq!(&f[6..10], &[1., 4., 0., 0.]);
        assert_eq!(&f[18..22], &[0.5, 0.25, -1e4, -1e4]);
    }

    #[test]
    fn tokens_use_the_word_separator_and_id_order() {
        let v = json!({ "ava": { "б": 5, "<pad>": 0, "|": 4, "а": 6 } });
        assert_eq!(tokens_for(&v, "ava").unwrap(), "<pad> 0\n| 4\nб 5\nа 6\n");
    }

    #[test]
    fn the_language_list_knows_avar_and_dargwa() {
        assert_eq!(name_of("ava"), Some("Avar (Avaric)"));
        assert_eq!(name_of("dar"), Some("Dargwa"));
        assert!(languages().len() > 1000);
    }
}
