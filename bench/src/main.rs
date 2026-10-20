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
    time::{Duration, Instant},
};

use local::{
    models, runtime,
    speedtest::{machine, round3, run_child, unix_now, MMS_LANGS},
    Models,
};
use serde_json::{json, Value};

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Lana's bundle identifier, which names its data folder on every platform.
const APP_ID: &str = "io.github.threemmc.lana";

const QUICK: &[&str] = &["zipa-small", "whisper-base", "speakers"];
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--child") {
        let compute = std::env::var("LANA_BENCH_COMPUTE").ok().and_then(|c| runtime::Compute::parse(&c)).unwrap_or(runtime::Compute::Auto);
        println!("{}", local::speedtest::measure(&args[2], Path::new(&args[3]), compute));
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
        let r = run_child(&exe, "--child", id, dir, *compute, None, &std::sync::atomic::AtomicBool::new(false));
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


fn ask(q: &str) -> String {
    print!("{q}");
    let _ = std::io::stdout().flush();
    let mut s = String::new();
    let _ = std::io::stdin().lock().read_line(&mut s);
    s
}
