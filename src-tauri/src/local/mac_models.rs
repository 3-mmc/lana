// SPDX-License-Identifier: GPL-3.0-or-later
// The bundled Mac model runners. Each stays alive for a job, receives 16 kHz
// float32 audio and returns either logits or timed words. A native assertion
// ends that worker, not Lana; the caller can rerun the job on int8 ONNX.

use super::{frames::Mat, runtime};
use serde_json::Value;
use std::{
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

static SERIAL: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

pub fn peak_bytes() -> u64 {
    PEAK.load(Ordering::Relaxed)
}

pub fn minimum_os(id: &str) -> Option<u32> {
    match id {
        "parakeet-tdt-0.6b-v3" => Some(14),
        "omnilingual-300m" | "omnilingual-1b" => Some(15),
        _ => None,
    }
}

fn helper(id: &str) -> Option<PathBuf> {
    let name = match id {
        "parakeet-tdt-0.6b-v3" => "parakeet/lana-parakeet",
        "omnilingual-300m" | "omnilingual-1b" => "omnilingual/lana-omnilingual",
        _ => return None,
    };
    let mut dirs = Vec::new();
    if let Some(p) = std::env::var_os("LANA_MAC_MODELS") {
        dirs.push(PathBuf::from(p));
    }
    if let Some(p) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        dirs.push(p.join("../Resources/mac-models"));
        dirs.push(p.join("mac-models"));
    }
    dirs.push(PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../src-tauri/mac-models"
    )));
    dirs.into_iter().map(|p| p.join(name)).find(|p| p.is_file())
}

pub fn available(id: &str) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && minimum_os(id).is_some_and(|minimum| {
            sysinfo::System::os_version()
                .and_then(|v| v.split('.').next()?.parse::<u32>().ok())
                .is_some_and(|major| major >= minimum)
        })
        && helper(id).is_some()
}

pub fn any_available() -> bool {
    available("parakeet-tdt-0.6b-v3") || available("omnilingual-300m")
}

pub fn model_id(path: &Path) -> Option<&'static str> {
    match path.parent()?.file_name()?.to_str()? {
        "mac-mlx-300m" => Some("omnilingual-300m"),
        "mac-mlx-1b" => Some("omnilingual-1b"),
        "mac-coreml" => Some("parakeet-tdt-0.6b-v3"),
        _ => None,
    }
}

pub struct Worker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    scratch: PathBuf,
    pub tokens: Option<Vec<String>>,
}

impl Worker {
    pub fn load(path: &Path) -> Result<Self, String> {
        let id = model_id(path).ok_or("not a Mac model")?;
        let executable =
            helper(id).ok_or("The Mac model runner is missing from this installation.")?;
        Self::spawn(path, id, &executable)
    }

    fn spawn(path: &Path, id: &str, executable: &Path) -> Result<Self, String> {
        let scratch = std::env::temp_dir().join(format!(
            "lana-mac-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let stderr =
            std::fs::File::create(scratch.join("stderr.txt")).map_err(|e| e.to_string())?;
        let mut command = Command::new(executable);
        command.arg(path.parent().ok_or("missing model directory")?);
        if id.starts_with("omnilingual") {
            command.arg(if id.ends_with("1b") { "1B" } else { "300M" });
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .map_err(|e| format!("could not start the Mac model runner: {e}"))?;
        let input = child.stdin.take().ok_or("missing model runner input")?;
        let output = BufReader::new(child.stdout.take().ok_or("missing model runner output")?);
        let mut worker = Self {
            child,
            input,
            output,
            scratch,
            tokens: None,
        };
        let greeting = worker.header()?;
        if let Some(tokens) = greeting.get("tokens") {
            let tokens: Vec<String> =
                serde_json::from_value(tokens.clone()).map_err(|e| e.to_string())?;
            if tokens.is_empty() || tokens.len() > 100_000 {
                return Err("invalid Mac model vocabulary".into());
            }
            worker.tokens = Some(tokens);
        }
        runtime::record(
            path,
            if id.starts_with("omnilingual") {
                "MLX (Metal, 8-bit)"
            } else {
                "CoreML (FluidAudio, CPU and Neural Engine)"
            },
        );
        Ok(worker)
    }

    fn failure(&mut self, error: impl std::fmt::Display) -> String {
        let _ = self.child.kill();
        let status = self.child.wait().ok();
        let log = std::fs::read_to_string(self.scratch.join("stderr.txt")).unwrap_or_default();
        // Keep the entire compiler error in the parent/benchmark stderr.
        eprintln!("lana: Mac model runner failed: {error} (exit: {status:?})\n{log}");
        format!(
            "Mac model runner failed: {error}. {}",
            log.lines()
                .rev()
                .find(|s| !s.trim().is_empty())
                .unwrap_or("No diagnostic output.")
        )
    }

    fn header(&mut self) -> Result<Value, String> {
        let mut line = String::new();
        // JSON headers contain at most a vocabulary or timed words. Bound a
        // broken worker's response before allocating a logits buffer.
        let read = (&mut self.output)
            .take(4 * 1024 * 1024)
            .read_line(&mut line);
        match read {
            Ok(0) => return Err(self.failure("worker exited before returning a result")),
            Err(e) => return Err(self.failure(e)),
            _ => {}
        }
        let value: Value = serde_json::from_str(&line).map_err(|e| self.failure(e))?;
        if value["protocol"] != "lana-mac-1" {
            return Err(self.failure("unsupported worker protocol"));
        }
        if let Some(peak) = value["peak_bytes"].as_u64() {
            PEAK.fetch_max(peak, Ordering::Relaxed);
        }
        Ok(value)
    }

    fn send(&mut self, samples: &[f32]) -> Result<Value, String> {
        if !(400..=640_000).contains(&samples.len()) {
            return Err("Mac model audio must be between 25 ms and 40 s.".into());
        }
        let mut bytes = Vec::with_capacity(8 + samples.len() * 4);
        bytes.extend_from_slice(&(samples.len() as u64).to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        self.input.write_all(&bytes).map_err(|e| self.failure(e))?;
        self.header()
    }

    pub fn logits(&mut self, samples: &[f32]) -> Result<Mat, String> {
        let header = self.send(samples)?;
        let (t, v) = dimensions(&header)?;
        if self.tokens.as_ref().map(Vec::len) != Some(v) {
            return Err(self.failure("logits do not match the vocabulary"));
        }
        let mut data = vec![0u8; t * v * 4];
        self.output
            .read_exact(&mut data)
            .map_err(|e| self.failure(e))?;
        let lp = data
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes(p.try_into().unwrap()))
            .collect();
        Ok(Mat { t, v, lp })
    }

    pub fn words(
        &mut self,
        samples: &[f32],
        offset: f64,
        end: f64,
    ) -> Result<Vec<super::parakeet::Word>, String> {
        let header = self.send(samples)?;
        let words = header["words"]
            .as_array()
            .ok_or("missing Mac model word timings")?;
        let mut out = Vec::new();
        for word in words {
            let text = word["text"].as_str().ok_or("missing word text")?;
            let a = word["start"].as_f64().ok_or("missing word start")? + offset;
            let b = word["end"].as_f64().ok_or("missing word end")? + offset;
            if !a.is_finite() || !b.is_finite() || a < offset || b < a || b > end + 0.1 {
                return Err("invalid Mac model word timings".into());
            }
            if !text.trim().is_empty() {
                out.push((text.trim().to_owned(), a.min(end), b.min(end)));
            }
        }
        Ok(out)
    }
}

fn dimensions(header: &Value) -> Result<(usize, usize), String> {
    let t = header["t"]
        .as_u64()
        .ok_or("missing logits time dimension")?;
    let v = header["v"]
        .as_u64()
        .ok_or("missing logits vocabulary dimension")?;
    if t == 0 || v == 0 || t > 2000 || v > 100_000 || t.saturating_mul(v) > 25_000_000 {
        return Err("invalid Mac model logits dimensions".into());
    }
    Ok((t as usize, v as usize))
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn invalid_logits_cannot_request_unbounded_memory() {
        assert_eq!(
            dimensions(&json!({"t":500,"v":10288})).unwrap(),
            (500, 10288)
        );
        for (t, v) in [
            (0, 10288),
            (500, 0),
            (2001, 10288),
            (2000, 100000),
            (u64::MAX, 2),
        ] {
            assert!(dimensions(&json!({"t":t,"v":v})).is_err());
        }
    }
    #[test]
    fn mac_model_paths_and_platform_requirements() {
        assert_eq!(
            model_id(Path::new(
                "models/omnilingual-1b/mac-mlx-1b/model.safetensors"
            )),
            Some("omnilingual-1b")
        );
        assert_eq!(
            model_id(Path::new("models/omnilingual-1b/model.fp32.onnx")),
            None
        );
        assert_eq!(minimum_os("parakeet-tdt-0.6b-v3"), Some(14));
        assert_eq!(minimum_os("omnilingual-300m"), Some(15));
        if !cfg!(target_os = "macos") {
            assert!(!any_available());
        }
    }

    #[cfg(unix)]
    #[test]
    fn worker_reads_logits_and_words_and_surfaces_a_native_crash() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!(
            "lana-worker-test-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let script = root.join("worker");
        std::fs::write(
            &script,
            r#"#!/usr/bin/env python3
import json, os, resource, signal, struct, sys
mode = sys.argv[1].split('/')[-1]
def header(v):
    v['protocol'] = 'lana-mac-1'
    sys.stdout.buffer.write((json.dumps(v)+'\n').encode())
    sys.stdout.buffer.flush()
if mode == 'crash':
    sys.stderr.write('error: contracting dimensions differ 1 & 1024\nassertion failed\n')
    sys.stderr.flush()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.kill(os.getpid(), signal.SIGABRT)
header({'tokens':['', 'a', ' '], 'backend':'test'})
while True:
    count = sys.stdin.buffer.read(8)
    if not count: break
    n = struct.unpack('<Q', count)[0]
    data = sys.stdin.buffer.read(n * 4)
    assert len(data) == n * 4
    if mode == 'words':
        header({'words':[{'text':'hello','start':0.0,'end':0.02}], 'peak_bytes':1048576})
    else:
        header({'t':2,'v':3, 'peak_bytes':1048576})
        sys.stdout.buffer.write(struct.pack('<6f',0,1,2,3,4,5))
        sys.stdout.buffer.flush()
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let samples = vec![0.0; 400];
        let mut worker =
            Worker::spawn(&root.join("logits/model"), "omnilingual-300m", &script).unwrap();
        assert_eq!(worker.tokens.as_ref().unwrap(), &["", "a", " "]);
        // Two requests on one process must not lose bytes between the JSON
        // header and binary payload in BufReader's buffer.
        for _ in 0..2 {
            let mat = worker.logits(&samples).unwrap();
            assert_eq!((mat.t, mat.v), (2, 3));
            assert_eq!(mat.lp, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        }
        drop(worker);
        let mut worker =
            Worker::spawn(&root.join("words/model"), "parakeet-tdt-0.6b-v3", &script).unwrap();
        assert_eq!(
            worker.words(&samples, 2.0, 2.025).unwrap(),
            vec![("hello".to_string(), 2.0, 2.02)]
        );
        drop(worker);
        let failure =
            match Worker::spawn(&root.join("crash/model"), "parakeet-tdt-0.6b-v3", &script) {
                Ok(_) => panic!("crashed worker was accepted"),
                Err(e) => e,
            };
        assert!(failure.contains("assertion failed"), "{failure}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
