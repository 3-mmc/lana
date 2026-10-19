// SPDX-License-Identifier: GPL-3.0-or-later
// lana-bench: time Lana's built-in recognisers on this computer.
//
// The numbers are for deciding which models Lana should recommend on which
// hardware, so they have to be the app's own numbers: the recognisers are
// src-tauri/src/local, included here by path rather than copied, and each
// model runs through the same Local::transcribe the app calls.
//
// Each model runs in a child process (this program again, with --child), so
// its peak memory is its own and a model that runs out of memory is recorded
// as a failure rather than ending the run.
//
// Nothing is uploaded. The result is one JSON file next to this program.

#[allow(dead_code)]
#[path = "../../src-tauri/src/local/mod.rs"]
mod local;
mod eval;

use std::{
    io::{BufRead, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use local::{models, runtime, Local, Models, Request};
use serde_json::{json, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Lana's bundle identifier, which names its data folder on every platform.
const APP_ID: &str = "io.github.threemmc.lana";

struct Clip {
    id: &'static str,
    file: &'static str,
    bytes: &'static [u8],
    /// Two-letter code for Whisper.
    lang: &'static str,
    reference: &'static str,
}

const CLIPS: &[Clip] = &[
    Clip {
        id: "ka", file: "ka-vinson.wav", lang: "ka",
        bytes: include_bytes!("../../web/demo/ka-vinson.wav"),
        reference: "ანტარქტიკის უმაღლესი მთები, ვინსონის მასივი, რომლის უმაღლესი მწვერვალი, ვინსონის მთა მდებარეობს 4892 მეტრზე, განლაგებულია სენტინელის ქედის ჩრდილოეთ ნაწილში.",
    },
    Clip {
        id: "de", file: "de-vinson.wav", lang: "de",
        bytes: include_bytes!("../../web/demo/de-vinson.wav"),
        reference: "Der nördliche Teil oder die Sentinel Range hat die höchsten Berge der Antarktis, das Vinson-Massiv, dessen höchster Gipfel mit 4892 m Höhe der Mount Vinson ist.",
    },
    Clip {
        id: "dar", file: "dar-48608859.mp3", lang: "",
        bytes: include_bytes!("../clips/dar-48608859.mp3"),
        reference: "ГӀяйшатлира ухьа бархьбатур, лацла бяхӀлизибси кьасиличиб липӀ-липӀбикӀуси чирагъла шала ахъбуциб.",
    },
];

const QUICK: &[&str] = &["zipa-small", "whisper-base", "speakers"];
/// MMS languages tested in the full run, one per clip it can read.
const MMS_LANGS: &[(&str, &str)] = &[("dar", "dar"), ("kat", "ka")];

/// Which clips a model is run on, a clip it reads well first. The first is run twice: once from a cold
/// start, once warm, so loading and steady speed can be told apart.
fn clips_for(model: &str) -> Vec<&'static Clip> {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        child(&args[2], Path::new(&args[3]));
        return;
    }
    let pause = !args.iter().any(|a| a == "--no-pause");
    let result = if args.iter().any(|a| a == "--eval") { run_eval(&args) } else { run(&args) };
    if let Err(e) = result {
        println!("\nlana-bench stopped: {e}");
    }
    if pause {
        print!("\nPress Enter to close.");
        let _ = std::io::stdout().flush();
        let _ = std::io::stdin().lock().read_line(&mut String::new());
    }
}

fn run(args: &[String]) -> Result<(), String> {
    println!("lana-bench {VERSION}: times Lana's built-in speech models on this computer.");
    println!("Nothing is uploaded. The results are saved as one file next to this program.\n");

    let lana_dir = app_data_dir().map(|d| d.join("models")).filter(|d| d.is_dir());
    let own_dir = cache_dir().join("lana-bench-models");
    let lana = lana_dir.as_ref().map(|d| Models::new(d.clone()));
    let own = Models::new(own_dir.clone());
    match &lana_dir {
        Some(d) => println!("Lana's models folder: {} (models already there are used, not downloaded again)", d.display()),
        None => println!("Lana's models folder was not found, so models are downloaded to {}", own_dir.display()),
    }

    // --models a,b: just those, by catalogue id (mms-<code> for MMS).
    let chosen: Option<Vec<String>> = args.iter().position(|a| a == "--models").and_then(|i| args.get(i + 1))
        .map(|l| l.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect());
    let full = if chosen.is_some() || args.iter().any(|a| a == "--full") {
        true
    } else if args.iter().any(|a| a == "--quick") {
        false
    } else {
        println!("\n  1  Quick: ZIPA small, Whisper base and Speakers. About 250 MB to download, a few minutes.");
        println!("  2  Full: every built-in model. About 4 GB to download unless Lana already has them,");
        println!("     and 10-30 minutes depending on the computer.");
        ask("\nWhich test? [1/2, default 1] ").trim() == "2"
    };

    let mut ids: Vec<String> = if let Some(c) = &chosen {
        c.clone()
    } else if full {
        models::catalogue().iter().map(|e| e.id.to_string()).collect()
    } else {
        QUICK.iter().map(|s| s.to_string()).collect()
    };
    if full && chosen.is_none() {
        let accepted_already = lana.as_ref().is_some_and(|m| m.mms_accepted());
        let include = args.iter().any(|a| a == "--mms") || accepted_already || {
            println!("\nMMS (Meta) is licensed CC BY-NC 4.0: non-commercial use only.");
            ask("Include MMS in the test, accepting that licence? [y/N] ").trim().eq_ignore_ascii_case("y")
        };
        if include {
            ids.extend(MMS_LANGS.iter().map(|(code, _)| format!("mms-{code}")));
        }
    }

    let (plan, fetched_any) = locate(&ids, lana.as_ref(), &own)?;

    // --compute cpu,auto: each model on each, so the accelerator can be
    // weighed against the CPU on the same computer. Auto by default, which
    // is what the app does.
    let computes: Vec<runtime::Compute> = match args.iter().position(|a| a == "--compute").and_then(|i| args.get(i + 1)) {
        Some(list) => list.split(',').map(|c| runtime::Compute::parse(c.trim()).ok_or(format!("--compute takes cpu, auto or gpu, not {c}")))
            .collect::<Result<_, _>>()?,
        None => vec![runtime::Compute::Auto],
    };

    let machine = machine();
    println!("\nThis computer: {} · {} · {} cores · {} GB memory",
             machine["os"].as_str().unwrap_or("?"), machine["cpu"].as_str().unwrap_or("?"),
             machine["logical_cores"], machine["memory_gb"]);
    println!("\n{:<28} {:>9} {:>9} {:>10} {:>8}", "model", "cold s", "warm s", "x realtime", "peak MB");

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut results = Vec::new();
    for ((id, dir), compute) in plan.iter().flat_map(|p| computes.iter().map(move |c| (p, c))) {
        let label = if computes.len() > 1 { format!("{id} [{}]", compute.name()) } else { id.clone() };
        print!("{label:<28} ");
        let _ = std::io::stdout().flush();
        let out = Command::new(&exe)
            .args(["--child", id, &dir.to_string_lossy()])
            .env("LANA_BENCH_COMPUTE", compute.name())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed = stdout.lines().rev().find_map(|l| serde_json::from_str::<Value>(l).ok());
        let r = match parsed {
            Some(v) => v,
            None => {
                let err = String::from_utf8_lossy(&out.stderr);
                let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("no output");
                json!({ "model": id, "compute": compute.name(), "ok": false, "exit": out.status.code(), "error": last })
            }
        };
        print_row(&r);
        results.push(r);
    }

    let report = json!({
        "tool": "lana-bench", "version": VERSION,
        "when": unix_now(), "mode": if full { "full" } else { "quick" },
        "machine": machine,
        "models": results,
    });
    let name = format!("lana-bench-{}-{}.json", std::env::consts::OS, unix_now());
    let dest = exe.parent().map(|p| p.join(&name)).filter(|p| std::fs::write(p, b"").is_ok())
        .or_else(|| home_dir().map(|h| h.join(&name)))
        .ok_or("nowhere to save the results")?;
    std::fs::write(&dest, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    println!("\nSaved: {}", dest.display());
    println!("Send that file back; it holds the timings and this computer's processor, memory and system, nothing else.");

    if fetched_any && !args.iter().any(|a| a == "--keep") {
        let mb = dir_size(&own_dir) / 1_048_576;
        if !ask(&format!("\nDelete the {mb} MB of models this test downloaded? [Y/n] ")).trim().eq_ignore_ascii_case("n") {
            let _ = std::fs::remove_dir_all(&own_dir);
            println!("Deleted.");
        }
    }
    Ok(())
}

/// --eval <folder>: boundary timing against phone-aligned references (see eval.rs).
fn run_eval(args: &[String]) -> Result<(), String> {
    println!("lana-bench {VERSION}: boundary timing against a phone-aligned reference.\n");
    let lana = app_data_dir().map(|d| d.join("models")).filter(|d| d.is_dir()).map(Models::new);
    let own = Models::new(cache_dir().join("lana-bench-models"));
    let started = Instant::now();
    let results = eval::run(args, |ids| locate(ids, lana.as_ref(), &own).map(|(p, _)| p))?;
    let report = json!({
        "tool": "lana-bench", "version": VERSION, "when": unix_now(), "mode": "eval",
        "seconds": round3(started.elapsed().as_secs_f64()),
        "machine": machine(),
        "eval": results,
    });
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let name = format!("lana-bench-eval-{}-{}.json", std::env::consts::OS, unix_now());
    let dest = exe.parent().map(|p| p.join(&name)).filter(|p| std::fs::write(p, b"").is_ok())
        .or_else(|| home_dir().map(|h| h.join(&name)))
        .ok_or("nowhere to save the results")?;
    std::fs::write(&dest, serde_json::to_string_pretty(&report).unwrap()).map_err(|e| e.to_string())?;
    println!("\nSaved: {}", dest.display());
    Ok(())
}

/// Where each model will come from: Lana's folder if it has it, ours
/// otherwise, downloading what neither has; and whether anything was fetched.
fn locate(ids: &[String], lana: Option<&Models>, own: &Models) -> Result<(Vec<(String, PathBuf)>, bool), String> {
    let mut plan: Vec<(String, PathBuf)> = Vec::new();
    let mut fetched_any = false;
    for id in ids {
        if let Some(m) = lana.filter(|m| has(m, id)) {
            plan.push((id.clone(), m.dir.clone()));
            continue;
        }
        if !has(own, id) {
            fetch(own, id)?;
            fetched_any = true;
        }
        plan.push((id.clone(), own.dir.clone()));
    }
    // The VAD every recogniser cuts its pieces with comes down with a model
    // download; a folder whose models are all there already may lack it.
    let mut dirs: Vec<&PathBuf> = plan.iter().map(|(_, d)| d).collect();
    dirs.dedup();
    for d in dirs {
        let m = Models::new(d.clone());
        if !m.vad_path().is_file() {
            let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
            if let Err(e) = rt.block_on(m.fetch_vad()) {
                println!("Could not download the VAD model ({e}); pieces are cut by energy instead.");
            }
        }
    }
    Ok((plan, fetched_any))
}

fn print_row(r: &Value) {
    if r["ok"] != json!(true) {
        println!("failed: {}", r["error"].as_str().unwrap_or("?"));
        return;
    }
    let f = |v: &Value| v.as_f64().map_or("-".into(), |x| format!("{x:.2}"));
    println!("{:>9} {:>9} {:>10} {:>8}", f(&r["cold_seconds"]), f(&r["warm_seconds"]),
             r["x_realtime"].as_f64().map_or("-".into(), |x| format!("{x:.1}x")),
             r["peak_mb"].as_u64().map_or("-".into(), |x| x.to_string()));
}

fn has(m: &Models, id: &str) -> bool {
    match id.strip_prefix("mms-") {
        Some(code) => m.mms_lang_installed(code),
        None => models::find(id).is_some_and(|e| m.is_installed(&e)),
    }
}

/// Download one model with the app's own downloader, showing progress.
fn fetch(m: &Models, id: &str) -> Result<(), String> {
    let rt = tokio::runtime::Runtime::new().map_err(|e| e.to_string())?;
    let _guard = rt.enter();
    let mms = id.strip_prefix("mms-");
    match mms {
        Some(code) => {
            m.mms_accept()?;
            m.mms_start(code)?;
        }
        None => m.start(id)?,
    }
    let label = id.to_string();
    loop {
        rt.block_on(tokio::time::sleep(Duration::from_millis(500)));
        let (done, total, err) = match mms {
            Some(code) => {
                let s = m.mms_status();
                let d = &s["downloading"][code];
                let sh = &s["shared"]["downloading"];
                let (done, total) = if !sh.is_null() { (sh["done"].as_u64(), sh["total"].as_u64()) } else { (d["done"].as_u64(), d["total"].as_u64()) };
                (done, total, s["errors"][code].as_str().map(String::from))
            }
            None => {
                let l = m.list();
                let e = l["models"].as_array().and_then(|a| a.iter().find(|e| e["id"] == id)).cloned().unwrap_or_default();
                (e["downloading"]["done"].as_u64(), e["downloading"]["total"].as_u64(), e["error"].as_str().map(String::from))
            }
        };
        if let Some(e) = err {
            println!();
            return Err(format!("could not download {label}: {e}"));
        }
        if has(m, id) {
            println!("\rDownloading {label}: done.{:30}", "");
            return Ok(());
        }
        if let (Some(d), Some(t)) = (done, total) {
            if t > 0 {
                print!("\rDownloading {label}: {} of {} MB   ", d / 1_048_576, t / 1_048_576);
                let _ = std::io::stdout().flush();
            }
        }
    }
}

/// One model over its clips, in this process. Prints one JSON line.
fn child(id: &str, dir: &Path) {
    let local = Local::new(Models::new(dir.to_path_buf()));
    // After Local::new, which reads the app's own setting from the folder.
    let compute = std::env::var("LANA_BENCH_COMPUTE").ok().and_then(|c| runtime::Compute::parse(&c)).unwrap_or(runtime::Compute::Auto);
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
    let result = json!({
        "model": id,
        "compute": compute.name(),
        // The runtime, the accelerator, and where each of the model's files ran.
        "runtime": runtime::describe(),
        "ok": error.is_none(),
        "error": error,
        "cold_seconds": runs.first().map(|r| r["seconds"].clone()),
        "warm_seconds": warm.first().map(|r| r["seconds"].clone()),
        "x_realtime": (secs > 0.0).then(|| round3(audio / secs)),
        "peak_mb": peak_rss() / 1_048_576,
        "vad": local.models.vad_path().is_file(),
        "runs": runs,
    });
    println!("{result}");
}

/// Character error rate, after lowercasing and dropping punctuation, so the
/// numbers compare spelling rather than formatting. Digits are kept: a model
/// that writes "4892" as words is charged for it, the same on every computer.
fn cer(reference: &str, hyp: &str, palochka: bool) -> f64 {
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

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn machine() -> Value {
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
        "whisper_backend": if cfg!(target_os = "macos") { "metal" } else { "cpu" },
    })
}

/// This process's peak resident memory, in bytes.
#[cfg(target_os = "linux")]
fn peak_rss() -> u64 {
    std::fs::read_to_string("/proc/self/status").ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).map(String::from))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|k| k.parse::<u64>().ok()))
        .map_or(0, |kb| kb * 1024)
}

#[cfg(target_os = "macos")]
fn peak_rss() -> u64 {
    // ru_maxrss is in bytes on macOS (kilobytes on Linux).
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut u) };
    u.ru_maxrss as u64
}

#[cfg(windows)]
fn peak_rss() -> u64 {
    use windows_sys::Win32::System::{ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS}, Threading::GetCurrentProcess};
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PeakWorkingSetSize as u64
}

/// Lana's data folder, as Tauri's app_data_dir names it on each platform.
fn app_data_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("APPDATA").map(|p| PathBuf::from(p).join(APP_ID))
    } else if cfg!(target_os = "macos") {
        home_dir().map(|h| h.join("Library/Application Support").join(APP_ID))
    } else {
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
            .or_else(|| home_dir().map(|h| h.join(".local/share")))
            .map(|d| d.join(APP_ID))
    }
}

/// Where downloaded models go when Lana has not got them. Not the temp
/// folder: on many Linux systems /tmp is in memory, and the full test is 4 GB.
fn cache_dir() -> PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from)
    } else if cfg!(target_os = "macos") {
        home_dir().map(|h| h.join("Library/Caches"))
    } else {
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).or_else(|| home_dir().map(|h| h.join(".cache")))
    };
    base.unwrap_or_else(std::env::temp_dir)
}

fn home_dir() -> Option<PathBuf> {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn dir_size(p: &Path) -> u64 {
    std::fs::read_dir(p).map_or(0, |rd| rd.flatten().map(|e| {
        let m = e.metadata().ok();
        if m.as_ref().is_some_and(|m| m.is_dir()) { dir_size(&e.path()) } else { m.map_or(0, |m| m.len()) }
    }).sum())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn ask(q: &str) -> String {
    print!("{q}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().lock().read_line(&mut s);
    s
}
