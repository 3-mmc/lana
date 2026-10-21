// SPDX-License-Identifier: GPL-3.0-or-later
// The speed test: how fast each built-in model runs on this computer, on the
// processor and on the GPU or Neural Engine, for deciding which models Lana
// should recommend on which hardware. The numbers have to be the app's own,
// so each model runs through the same Local::transcribe a user's job does.
//
// Each model runs in a child process (this program again, with
// --speed-test-child), so its peak memory is its own and a model that runs
// out of memory is recorded as a failure rather than ending the test. Both
// lana-bench (the command-line tool in bench/) and the app's settings run it.
//
// The full report stays local. Sharing uses a separate allowlisted report
// and requires explicit one-time consent in the app.

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use serde_json::{json, Value};

use super::{models, runtime, Local, Models, Request};

pub struct Clip {
    pub id: &'static str,
    pub file: &'static str,
    pub bytes: &'static [u8],
    /// Two-letter code for Whisper.
    pub lang: &'static str,
    pub reference: &'static str,
}

pub const CLIPS: &[Clip] = &[
    Clip {
        id: "ka", file: "ka-vinson.wav", lang: "ka",
        bytes: include_bytes!("../../../web/demo/ka-vinson.wav"),
        reference: "ანტარქტიკის უმაღლესი მთები, ვინსონის მასივი, რომლის უმაღლესი მწვერვალი, ვინსონის მთა მდებარეობს 4892 მეტრზე, განლაგებულია სენტინელის ქედის ჩრდილოეთ ნაწილში.",
    },
    Clip {
        id: "de", file: "de-vinson.wav", lang: "de",
        bytes: include_bytes!("../../../web/demo/de-vinson.wav"),
        reference: "Der nördliche Teil oder die Sentinel Range hat die höchsten Berge der Antarktis, das Vinson-Massiv, dessen höchster Gipfel mit 4892 m Höhe der Mount Vinson ist.",
    },
    Clip {
        id: "dar", file: "dar-48608859.mp3", lang: "",
        bytes: include_bytes!("../../../bench/clips/dar-48608859.mp3"),
        reference: "ГӀяйшатлира ухьа бархьбатур, лацла бяхӀлизибси кьасиличиб липӀ-липӀбикӀуси чирагъла шала ахъбуциб.",
    },
];

/// MMS languages tested, one per clip it can read.
pub const MMS_LANGS: &[(&str, &str)] = &[("dar", "dar"), ("kat", "ka")];

/// Which clips a model is run on, a clip it reads well first. The first is run twice: once from a cold
/// start, once warm, so loading and steady speed can be told apart.
pub fn clips_for(model: &str) -> Vec<&'static Clip> {
    let ids: &[&str] = match model {
        m if m.starts_with("whisper") => &["de", "ka"],
        // Parakeet knows no Georgian; the ka run shows what it does instead.
        "parakeet-tdt-0.6b-v3" => &["de", "ka"],
        "sounds" => &["de", "ka"],
        "mms-dar" => &["dar"],
        "mms-kat" => &["ka"],
        "speakers" => &["de", "ka"],
        _ => &["de", "ka", "dar"],
    };
    ids.iter().map(|id| CLIPS.iter().find(|c| c.id == *id).unwrap()).collect()
}

/// One model over its clips, in this process, with `compute` as the
/// setting: the child's half of the test.
pub fn measure(id: &str, dir: &Path, compute: runtime::Compute) -> Value {
    let local = Local::new(Models::new(dir.to_path_buf()));
    // After Local::new, which reads the app's own setting from the folder.
    runtime::set_compute(compute);
    let clips = clips_for(id);
    let mut runs = Vec::new();
    let mut error = None;
    // The first clip twice: cold, then warm.
    let order: Vec<(&Clip, bool)> = std::iter::once((clips[0], true)).chain(clips.iter().map(|c| (*c, false))).collect();
    // LANA_BENCH_FILE reads that recording in place of each clip, to try a
    // model on other audio (with LANA_BENCH_DUMP to see what it returned).
    let own = std::env::var_os("LANA_BENCH_FILE").map(PathBuf::from);
    for (clip, cold) in order {
        let t = Instant::now();
        let (code, v) = local.transcribe(Request {
            model: id.into(),
            bytes: own.as_ref().map_or_else(|| clip.bytes.to_vec(), |p| std::fs::read(p).unwrap_or_default()),
            filename: own.as_ref().map_or_else(|| clip.file.into(), |p| p.to_string_lossy().into_owned()),
            language: (!clip.lang.is_empty() && id.starts_with("whisper")).then(|| clip.lang.into()),
            speakers: None,
            threshold: None,
        });
        let secs = t.elapsed().as_secs_f64();
        if code != 200 {
            error = Some(v["error"].as_str().unwrap_or("failed").to_string());
            break;
        }
        let audio = v["duration"].as_f64().unwrap_or(0.0);
        let text = v["text"].as_str().unwrap_or("").to_string();
        if std::env::var_os("LANA_BENCH_DUMP").is_some() {
            eprintln!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
        }
        let words = id.starts_with("whisper") || id.starts_with("omnilingual") || id.starts_with("mms-") || id.starts_with("parakeet");
        runs.push(json!({
            "clip": clip.id, "cold": cold, "audio_seconds": audio, "seconds": round3(secs),
            "x_realtime": round3(audio / secs.max(1e-6)),
            "cer": words.then(|| round3(cer(clip.reference, &text, clip.id == "dar"))),
            "speakers": v.get("speakers"),
            "warnings": v.get("warnings").filter(|w| w.as_array().is_some_and(|a| !a.is_empty())),
            "text": if id == "speakers" { Value::Null } else { json!(text) },
        }));
    }
    // The headline speed is the warm run of the first clip, which is always
    // one the model can read. A model given a language it does not know can
    // loop on its own output (Whisper small spent 42 s on 15 s of Georgian),
    // which says nothing about the hardware; those runs stay in `runs`.
    let warm: Vec<&Value> = runs.iter().filter(|r| r["cold"] == json!(false)).collect();
    let (audio, secs) = warm.first().map_or((0.0, 0.0), |r| (r["audio_seconds"].as_f64().unwrap_or(0.0), r["seconds"].as_f64().unwrap_or(0.0)));
    json!({
        "model": id,
        "compute": compute.name(),
        // The runtime, the accelerator, and where each of the model's files ran.
        "runtime": runtime::describe(),
        "ok": error.is_none(),
        "error": error,
        "cold_seconds": runs.first().map(|r| r["seconds"].clone()),
        "warm_seconds": warm.first().map(|r| r["seconds"].clone()),
        "x_realtime": (secs > 0.0).then(|| round3(audio / secs)),
        // Native Mac models run in a separate process. Report both process
        // high-water marks and their conservative sum, not just the broker.
        "peak_mb": (peak_rss() + super::mac_models::peak_bytes()) / 1_048_576,
        "parent_peak_mb": peak_rss() / 1_048_576,
        "native_peak_mb": super::mac_models::peak_bytes() / 1_048_576,
        "memory_method": if super::mac_models::peak_bytes() > 0 { "sum of process high-water marks" } else { "process high-water mark" },
        "vad": local.models.vad_path().is_file(),
        "runs": runs,
    })
}

/// Run one model in a child process and read its result. `exe` is a program
/// that answers `<child_flag> <id> <dir>` with `measure`'s JSON as its last
/// line of output. `cancel` stops it early.
pub fn run_child(exe: &Path, child_flag: &str, id: &str, dir: &Path, compute: runtime::Compute,
                 library: Option<&Path>, cancel: &AtomicBool) -> Value {
    let failed = |e: String| json!({ "model": id, "compute": compute.name(), "ok": false, "error": e });
    // Output goes to files, not pipes: whisper.cpp logs enough to fill a
    // pipe nobody is reading yet, which would stall the child.
    let scratch = std::env::temp_dir().join(format!("lana-speed-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&scratch);
    let (out_path, err_path) = (scratch.join("out.txt"), scratch.join("err.txt"));
    let (Ok(out), Ok(err)) = (std::fs::File::create(&out_path), std::fs::File::create(&err_path)) else {
        return failed("could not write the test's scratch files".into());
    };
    let mut cmd = Command::new(exe);
    cmd.args([child_flag, id, &dir.to_string_lossy()])
        .env("LANA_BENCH_COMPUTE", compute.name())
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err);
    // The same runtime as this process, the GPU pack included.
    if let Some(l) = library {
        cmd.env("LANA_ORT_LIBRARY", l);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return failed(format!("could not start the test: {e}")),
    };
    let status = loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return failed("stopped".into());
        }
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return failed(e.to_string()),
        }
    };
    let stdout = std::fs::read_to_string(&out_path).unwrap_or_default();
    let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
    let _ = std::fs::remove_dir_all(&scratch);
    child_result(id, compute, &stdout, &stderr, status)
}

fn child_result(id: &str, compute: runtime::Compute, stdout: &str, stderr: &str, status: std::process::ExitStatus) -> Value {
    let mut result = match stdout.lines().rev().filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["model"].as_str() == Some(id) && v["ok"].is_boolean()) {
        Some(v) if status.success() => v,
        _ => {
            let last = stderr.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no output");
            let mut v = json!({ "model": id, "compute": compute.name(), "ok": false, "error": last });
            v["exit"] = json!(status.code());
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                v["signal"] = json!(status.signal());
            }
            v["stdout"] = json!(stdout);
            v
        }
    };
    // The actual compiler error precedes the final abort assertion. Keep all
    // stderr, including successful runs that fell back from an accelerator.
    result["stderr"] = json!(stderr);
    result
}

/// Character error rate, after lowercasing and dropping punctuation, so the
/// numbers compare spelling rather than formatting. Digits are kept: a model
/// that writes "4892" as words is charged for it, the same on every computer.
pub fn cer(reference: &str, hyp: &str, palochka: bool) -> f64 {
    let norm = |s: &str| {
        let s: String = s.to_lowercase().chars().map(|c| {
            let c = if palochka && matches!(c, 'ӏ' | 'Ӏ' | 'i' | 'l' | '1') { 'ӏ' } else { c };
            if c.is_alphanumeric() || c == 'ӏ' { c } else { ' ' }
        }).collect();
        s.split_whitespace().collect::<Vec<_>>().join(" ").chars().collect::<Vec<char>>()
    };
    let (r, h) = (norm(reference), norm(hyp));
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for (i, rc) in r.iter().enumerate() {
        let mut cur = vec![i + 1; h.len() + 1];
        for (j, hc) in h.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(rc != hc)).min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[h.len()] as f64 / r.len().max(1) as f64
}

pub fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// The computer, as far as the numbers depend on it. Nothing that names the
/// user or the machine.
pub fn machine() -> Value {
    use sysinfo::System;
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpu = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
    json!({
        "os": System::long_os_version().unwrap_or_else(|| std::env::consts::OS.into()),
        "kernel": System::kernel_version(),
        "arch": std::env::consts::ARCH,
        "cpu": cpu,
        "logical_cores": sys.cpus().len(),
        "physical_cores": System::physical_core_count(),
        "memory_gb": ((sys.total_memory() as f64 / 1_073_741_824.0) * 10.0).round() / 10.0,
        "available_memory_gb": ((sys.available_memory() as f64 / 1_073_741_824.0) * 10.0).round() / 10.0,
        // Lana leaves one core free for the interface; see local::threads().
        "threads_used": std::thread::available_parallelism().map_or(4, |n| n.get().saturating_sub(1).clamp(1, 8)),
        "whisper_backend": if cfg!(target_os = "macos") { "metal" } else if cfg!(feature = "whisper-vulkan") { "vulkan" } else { "cpu" },
    })
}

/// This process's peak resident memory, in bytes.
#[cfg(target_os = "linux")]
pub fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(String::from))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|k| k.parse::<u64>().ok()))
        .map_or(0, |kb| kb * 1024)
}

#[cfg(target_os = "macos")]
pub fn peak_rss() -> u64 {
    // ru_maxrss is in bytes on macOS (kilobytes on Linux).
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    u.ru_maxrss as u64
}

#[cfg(windows)]
pub fn peak_rss() -> u64 {
    use windows_sys::Win32::System::{ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS}, Threading::GetCurrentProcess};
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PeakWorkingSetSize as u64
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

// ------------------------------------------------------------- in the app

/// The flag the app answers as a child of the test.
pub const CHILD_FLAG: &str = "--speed-test-child";

/// The test the settings screen runs: the models already downloaded, each
/// on the processor, and again on the accelerator where it has a version
/// for one. Nothing is downloaded for it.
pub fn plan(m: &Models) -> Vec<(String, runtime::Compute)> {
    let accelerated = runtime::has_accelerator();
    let mut out = Vec::new();
    for e in m.installed() {
        if e.kind == models::Kind::Pitch {
            continue; // fills no tier; transcribe does not run it
        }
        out.push((e.id.to_string(), runtime::Compute::Cpu));
        if (accelerated && m.variant(&e).is_some_and(|v| v.files.iter().all(|f| m.file(&e, f).is_file())))
            || (cfg!(target_os = "macos") && e.kind == models::Kind::Whisper) {
            out.push((e.id.to_string(), runtime::Compute::Auto));
        }
    }
    for (code, _) in MMS_LANGS {
        if m.mms_lang_installed(code) {
            out.push((format!("mms-{code}"), runtime::Compute::Cpu));
        }
    }
    out
}

#[derive(Default)]
pub struct Status {
    pub running: bool,
    pub total: usize,
    pub current: Option<String>,
    pub rows: Vec<Value>,
    /// Where the finished report was saved.
    pub file: Option<PathBuf>,
    pub report: Option<Value>,
    pub error: Option<String>,
}

#[derive(Clone, Default)]
pub struct Test {
    pub status: Arc<Mutex<Status>>,
    cancel: Arc<AtomicBool>,
}

impl Test {
    /// Start in the background; poll `status`. `save_in` is where the report
    /// goes (the Downloads folder).
    pub fn start(&self, models: Models, save_in: PathBuf) -> Result<(), String> {
        let plan = plan(&models);
        if plan.is_empty() {
            return Err("Download a model first: the test measures the models Lana has.".into());
        }
        {
            let mut s = self.status.lock().unwrap();
            if s.running {
                return Ok(());
            }
            *s = Status { running: true, total: plan.len(), ..Default::default() };
        }
        self.cancel.store(false, Ordering::Relaxed);
        let me = self.clone();
        std::thread::spawn(move || {
            let exe = std::env::current_exe();
            let library = runtime::library_path();
            let started = Instant::now();
            for (id, compute) in &plan {
                if me.cancel.load(Ordering::Relaxed) {
                    break;
                }
                me.status.lock().unwrap().current = Some(format!("{id} [{}]", compute.name()));
                let r = match &exe {
                    Ok(exe) => run_child(exe, CHILD_FLAG, id, &models.dir, *compute, library.as_deref(), &me.cancel),
                    Err(e) => json!({ "model": id, "ok": false, "error": e.to_string() }),
                };
                me.status.lock().unwrap().rows.push(r);
            }
            let mut s = me.status.lock().unwrap();
            s.running = false;
            s.current = None;
            if me.cancel.load(Ordering::Relaxed) {
                s.error = Some("Stopped.".into());
                return;
            }
            let report = json!({
                "tool": "lana", "version": env!("CARGO_PKG_VERSION"), "when": unix_now(), "mode": "in-app",
                "seconds": round3(started.elapsed().as_secs_f64()),
                "machine": machine(),
                "runtime": runtime::describe(),
                "models": s.rows,
            });
            let name = format!("lana-speed-test-{}-{}.json", std::env::consts::OS, unix_now());
            let dest = save_in.join(name);
            match std::fs::write(&dest, serde_json::to_string_pretty(&report).unwrap_or_default()) {
                Ok(()) => s.file = Some(dest),
                Err(e) => s.error = Some(format!("Could not save the results in {}: {e}", save_in.display())),
            }
            s.report = Some(report);
        });
        Ok(())
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn json(&self) -> Value {
        let s = self.status.lock().unwrap();
        json!({
            "running": s.running, "total": s.total, "done": s.rows.len(), "current": s.current,
            "rows": s.rows.iter().map(row).collect::<Vec<_>>(),
            "file": s.file.as_ref().map(|f| f.display().to_string()),
            "error": s.error,
            "report_id": s.report.as_ref().and_then(|r| r.get("when")),
        })
    }
}

/// A result row as the settings screen shows it.
fn row(r: &Value) -> Value {
    json!({
        "model": r["model"], "compute": r["compute"], "ok": r["ok"], "error": r["error"],
        "warm_seconds": r["warm_seconds"], "x_realtime": r["x_realtime"], "peak_mb": r["peak_mb"],
        "on": r["runtime"]["models"].as_object().map(|m| {
            let mut on: Vec<&str> = m.values().filter_map(Value::as_str).collect();
            on.sort();
            on.dedup();
            on.join(", ")
        }),
    })
}

/// The deliberately smaller report offered for sharing. Keep this an allowlist:
/// the local report also contains transcripts, paths and complete compiler logs.
/// Adding a diagnostic to the local report must never add it to a submission.
pub fn shared_report(report: &Value) -> Value {
    fn fields(value: &Value, names: &[&str]) -> Value {
        let object: serde_json::Map<String, Value> = names.iter()
            .filter_map(|name| value.get(*name).filter(|v| v.is_string() || v.is_number() || v.is_boolean()).map(|v| ((*name).into(), v.clone())))
            .collect();
        Value::Object(object)
    }
    let mut shared = fields(report, &["tool", "version", "seconds"]);
    shared["schema"] = json!(1);
    shared["machine"] = fields(&report["machine"], &[
        "os", "arch", "cpu", "logical_cores", "physical_cores", "memory_gb", "threads_used",
    ]);
    shared["runtime"] = fields(&report["runtime"], &["accelerator"]);
    shared["models"] = json!(report["models"].as_array().into_iter().flatten().map(|r| {
        let mut result = fields(r, &[
            "model", "compute", "ok", "cold_seconds", "warm_seconds", "x_realtime",
            "peak_mb", "parent_peak_mb", "native_peak_mb", "memory_method", "vad", "exit_code", "signal",
        ]);
        result["on"] = row(r)["on"].clone();
        result
    }).collect::<Vec<_>>());
    shared
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn child_abort_preserves_full_compiler_error_and_signal() {
        use std::os::unix::process::ExitStatusExt;
        let stderr = "error: contracting dimensions differ 1 & 1024\nnote: tensor<1x1x1xf32>\nfailed assertion\n";
        let v = child_result("omnilingual-300m", runtime::Compute::Auto, "", stderr, std::process::ExitStatus::from_raw(6));
        assert_eq!(v["ok"], false);
        assert_eq!(v["signal"], 6);
        assert_eq!(v["stderr"], stderr);
        assert_eq!(v["error"], "failed assertion");
    }

    #[cfg(unix)]
    #[test]
    fn successful_cpu_fallback_keeps_its_diagnostics() {
        use std::os::unix::process::ExitStatusExt;
        let stdout = "log line\n{\"model\":\"zipa-large\",\"ok\":true}\n";
        let v = child_result("zipa-large", runtime::Compute::Auto, stdout, "accelerator refused model", std::process::ExitStatus::from_raw(0));
        assert_eq!(v["ok"], true);
        assert_eq!(v["stderr"], "accelerator refused model");
    }

    #[test]
    fn cer_ignores_case_and_punctuation() {
        assert_eq!(cer("Der Teil, die Range.", "der teil die range", false), 0.0);
        assert!((cer("abc", "abd", false) - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn shared_report_omits_transcripts_paths_identifiers_and_logs() {
        let report = json!({
            "tool": "lana", "version": "0.2.1", "when": 12345, "seconds": 8.0,
            "hostname": "PRIVATE", "email": "PRIVATE", "future_diagnostic": "PRIVATE",
            "machine": { "os": "macOS 26", "cpu": "Apple M1 Pro", "arch": "aarch64", "logical_cores": 10,
                "memory_gb": 32.0, "hostname": "PRIVATE", "username": "PRIVATE", "kernel": "PRIVATE" },
            "runtime": { "library": "/Users/PRIVATE/runtime.dylib", "accelerator": "Apple Metal", "error": "PRIVATE" },
            "models": [{ "model": "whisper-base", "compute": "auto", "ok": false, "signal": 6,
                "warm_seconds": 0.4, "x_realtime": 27.0, "peak_mb": 364, "stderr": "PRIVATE", "stdout": "PRIVATE",
                "error": "PRIVATE", "runtime": { "library": "PRIVATE", "models": { "/Users/PRIVATE/model": "CPU" } },
                "runs": [{ "filename": "PRIVATE", "text": "PRIVATE", "reference": "PRIVATE" }] }],
        });
        let shared = shared_report(&report);
        let text = serde_json::to_string(&shared).unwrap();
        assert!(!text.contains("PRIVATE"));
        assert!(!text.contains("12345"));
        assert_eq!(shared["machine"]["cpu"], "Apple M1 Pro");
        assert_eq!(shared["models"][0]["signal"], 6);
        assert_eq!(shared["models"][0]["warm_seconds"], 0.4);
        assert_eq!(shared["models"][0]["on"], "CPU");
        let mut changed_schema = report.clone();
        changed_schema["machine"]["cpu"] = json!({ "brand": "Apple M1 Pro", "future_private_field": "PRIVATE" });
        assert!(shared_report(&changed_schema)["machine"].get("cpu").is_none());
    }

}
