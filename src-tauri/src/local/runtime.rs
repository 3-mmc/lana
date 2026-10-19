// SPDX-License-Identifier: GPL-3.0-or-later
// The ONNX Runtime every built-in model but Whisper runs on, and the one place
// sessions are made, so which processor a model runs on is decided here.
//
// The runtime is a shared library Lana ships beside itself (ort's
// load-dynamic), not one linked in: so each platform gets the build with its
// accelerator, and a Linux computer with an NVIDIA card can swap in the CUDA
// build later (the GPU pack, in the models folder) without a new Lana.
//
//   macOS     onnxruntime 1.30 with CoreML: the Neural Engine and the GPU
//   Windows   onnxruntime 1.24 with DirectML: any DirectX 12 GPU
//             (1.24.4 is the last DirectML build Microsoft published)
//   Linux     onnxruntime 1.30, CPU; with the GPU pack, CUDA
//
// `ort` is built for API 24, the oldest of these, and newer runtimes still
// serve that API.

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

use ort::{environment::Environment, ep::ExecutionProviderDispatch, session::Session};
use serde_json::{json, Value};

/// Which processor the built-in models may use, a setting.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compute {
    /// The accelerator if there is one and the model loads on it, else the CPU.
    Auto,
    /// The CPU only.
    Cpu,
    /// The accelerator or an error: for measuring it (lana-bench).
    Gpu,
}

impl Compute {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(Compute::Auto),
            "cpu" => Some(Compute::Cpu),
            "gpu" => Some(Compute::Gpu),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Compute::Auto => "auto",
            Compute::Cpu => "cpu",
            Compute::Gpu => "gpu",
        }
    }
}

/// What a model is, as far as choosing a processor goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// A large network run on long inputs (the recognisers' encoders, the
    /// pitch and language models): worth a GPU or the Neural Engine, when
    /// it is in floating point (see `quantised`).
    Heavy,
    /// A small network run many times on short inputs (VAD steps, a
    /// transducer's decoder and joiner, the speaker models): the CPU, where
    /// a call costs least.
    Light,
}

struct State {
    compute: Compute,
    /// Folders to look for the library in, likeliest first.
    dirs: Vec<PathBuf>,
    /// Where compiled and optimised models are kept (CoreML's compiled
    /// models, the CPU's optimised graphs), so that work is done once.
    cache: Option<PathBuf>,
    /// Each model file loaded, and the processor it got.
    used: BTreeMap<String, &'static str>,
}

static STATE: Mutex<State> = Mutex::new(State { compute: Compute::Auto, dirs: Vec::new(), cache: None, used: BTreeMap::new() });

pub fn set_compute(c: Compute) {
    STATE.lock().unwrap().compute = c;
}

pub fn compute() -> Compute {
    STATE.lock().unwrap().compute
}

/// Another folder the library may be in (the app's resources, the GPU pack).
/// Only folders added before the first model loads count: the library is
/// loaded once per run.
pub fn look_in(dir: PathBuf) {
    let mut s = STATE.lock().unwrap();
    if !s.dirs.contains(&dir) {
        s.dirs.push(dir);
    }
}

/// A folder to look in before all others (the GPU pack).
pub fn prefer(dir: PathBuf) {
    let mut s = STATE.lock().unwrap();
    s.dirs.retain(|d| *d != dir);
    s.dirs.insert(0, dir);
}

/// Where compiled and optimised models may be kept.
pub fn cache_in(dir: PathBuf) {
    STATE.lock().unwrap().cache = Some(dir);
}

/// Where in the models folder the GPU pack goes (Linux with an NVIDIA card):
/// the CUDA build of the runtime and the CUDA libraries it needs.
pub const GPU_PACK: &str = "runtime/cuda";

#[cfg(target_os = "windows")]
const LIBRARY: &str = "onnxruntime.dll";
#[cfg(target_os = "macos")]
const LIBRARY: &str = "libonnxruntime.dylib";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const LIBRARY: &str = "libonnxruntime.so";

/// The library, from LANA_ORT_LIBRARY, the folders given to `look_in`, or
/// beside the executable (and in its lib/, Resources/ or Frameworks/).
fn library() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("LANA_ORT_LIBRARY").map(PathBuf::from) {
        return Some(p);
    }
    let mut dirs = STATE.lock().unwrap().dirs.clone();
    // A development run, and lana-bench: where scripts/fetch-onnxruntime.sh
    // puts it.
    dirs.push(PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../src-tauri/onnxruntime")));
    if let Some(exe) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        dirs.push(exe.join("onnxruntime"));
        dirs.push(exe.clone());
        dirs.push(exe.join("lib"));
        dirs.push(exe.join("../Resources/onnxruntime"));
        dirs.push(exe.join("../Frameworks"));
        dirs.push(exe.join("../lib/lana-desktop/onnxruntime"));
    }
    dirs.into_iter().map(|d| d.join(LIBRARY)).find(|p| p.is_file())
}

/// Whether the library loaded is a CUDA build (the GPU pack): it carries the
/// CUDA provider beside it.
fn has_cuda(lib: &Path) -> bool {
    cfg!(target_os = "linux") && lib.with_file_name("libonnxruntime_providers_cuda.so").is_file()
}

struct Loaded {
    env: Environment,
    library: PathBuf,
    cuda: bool,
}

fn loaded() -> Result<&'static Loaded, String> {
    static LOADED: OnceLock<Result<Loaded, String>> = OnceLock::new();
    LOADED.get_or_init(|| {
        let library = library().ok_or_else(|| format!("ONNX Runtime ({LIBRARY}) is missing from this installation"))?;
        // onnxruntime.dll needs the Visual C++ runtime, which not every
        // Windows has installed, and DirectML.dll: both ship beside it. Windows
        // looks for a library's dependencies beside the executable, not beside
        // the library, so they are loaded first, from where they are.
        #[cfg(target_os = "windows")]
        preload_beside(&library, &["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "DirectML.dll"]);
        let env = ort::init_from(&library)
            .map_err(|e| format!("could not load ONNX Runtime from {}: {e}", library.display()))?
            .with_name("lana")
            // Windows builds of the runtime log usage events to Windows'
            // telemetry; Lana keeps its word about whom it talks to (the
            // privacy note in the settings).
            .with_telemetry(false)
            .build()
            .map_err(|e| format!("ONNX Runtime: {e}"))?;
        let cuda = has_cuda(&library);
        // The GPU pack keeps CUDA and cuDNN beside the runtime, where the CUDA
        // provider would not look for them: loaded now, they are found.
        #[cfg(target_os = "linux")]
        if cuda {
            preload_beside(&library, CUDA_LIBRARIES);
        }
        eprintln!("lana: ONNX Runtime from {}{}", library.display(), if cuda { " (CUDA)" } else { "" });
        Ok(Loaded { env, library, cuda })
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// What the CUDA 12 build of the runtime needs from CUDA and cuDNN 9, all
/// in the GPU pack. CUDA 12, not 13: 13 dropped the GTX 10xx cards (Pascal),
/// still common on researchers' desks.
#[cfg(target_os = "linux")]
const CUDA_LIBRARIES: &[&str] = &[
    "libcudart.so.12", "libcublasLt.so.12", "libcublas.so.12", "libcurand.so.10", "libcufft.so.11",
    "libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_heuristic.so.9", "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9", "libcudnn_engines_precompiled.so.9", "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_engines_tensor_ir.so.9", "libcudnn_ext.so.9",
];

/// Load these libraries from beside `library`, so that it and what it loads
/// later find them already loaded. Some need others of the list first; the
/// passes repeat until nothing more loads.
#[allow(dead_code)] // each platform uses it for something else
fn preload_beside(library: &Path, names: &[&str]) {
    let mut left: Vec<PathBuf> = names.iter().map(|n| library.with_file_name(n)).filter(|p| p.is_file()).collect();
    loop {
        let before = left.len();
        left.retain(|p| ort::util::preload_dylib(p.as_path()).is_err());
        if left.is_empty() || left.len() == before {
            break;
        }
    }
    for p in &left {
        eprintln!("lana: could not load {}", p.display());
    }
}

/// This computer's accelerator as an execution provider, if this build of
/// the runtime has one: its name for people, and the provider.
fn accelerator(l: &Loaded) -> Option<(&'static str, ExecutionProviderDispatch)> {
    #[cfg(target_os = "macos")]
    {
        use ort::ep::coreml::{ComputeUnits, ModelFormat};
        let mut ep = ort::ep::CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_compute_units(ComputeUnits::All);
        if let Some(dir) = STATE.lock().unwrap().cache.clone().map(|d| d.join("coreml")) {
            if std::fs::create_dir_all(&dir).is_ok() {
                ep = ep.with_model_cache_dir(dir.display());
            }
        }
        let _ = l;
        return Some(("CoreML (Neural Engine and GPU)", ep.build().error_on_failure()));
    }
    #[cfg(target_os = "windows")]
    {
        let _ = l;
        return Some(("DirectML (GPU)", ort::ep::DirectML::default().with_device_id(0).build().error_on_failure()));
    }
    #[cfg(target_os = "linux")]
    {
        return l.cuda.then(|| ("CUDA (NVIDIA GPU)", ort::ep::CUDA::default().with_device_id(0).build().error_on_failure()));
    }
    #[allow(unreachable_code)]
    {
        let _ = l;
        None
    }
}

/// Whether a model file is quantised to int8, by its name, or by the shared
/// part beside it for MMS (whose conversion is int8 throughout). Such a model
/// runs no faster on an accelerator than on the CPU, as most providers have
/// no kernels for dynamically quantised operators and pass each one back:
/// on a GTX 1080, ZIPA small took 0.20 s in int8 on CUDA, 0.18 s on the CPU,
/// and 0.07 s in fp32 on CUDA.
fn quantised(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    name.contains("int8") || name.contains("quantized") || path.with_file_name("mms-base.bin").exists()
}

/// Whether this computer has an accelerator Lana can use, whatever the
/// setting.
pub fn has_accelerator() -> bool {
    loaded().ok().and_then(accelerator).is_some()
}

/// Whether models in floating point would run on an accelerator now: there
/// is one, and the setting allows it. Then a model's float variant is
/// worth loading (models::Models::model_file).
pub fn wants_float() -> bool {
    compute() != Compute::Cpu && loaded().ok().and_then(accelerator).is_some()
}

/// Which precisions suit the accelerator, best first. CUDA: fp32, as
/// Pascal cards (GTX 10xx) run fp16 slower than fp32 (ZIPA small 0.15 s
/// against 0.07 s on a GTX 1080). CoreML and DirectML: fp16, which the
/// Neural Engine and most DirectX 12 GPUs run natively, at half the memory.
pub fn float_preference() -> &'static [&'static str] {
    if cfg!(target_os = "linux") { &["fp32", "fp16"] } else { &["fp16", "fp32"] }
}

/// A session for the model at `path`, on the accelerator when the setting
/// and the model's role allow and it loads there, else on the CPU. An int8
/// model stays on the CPU unless the setting is gpu (for measuring).
pub fn session(path: &Path, role: Role, threads: i32) -> Result<Session, String> {
    let l = loaded()?;
    let want = match compute() {
        _ if role == Role::Light => Compute::Cpu,
        Compute::Auto if quantised(path) => Compute::Cpu,
        c => c,
    };
    let fail = |e: ort::Error| format!("could not load {}: {e}", path.display());
    if want != Compute::Cpu {
        match accelerator(l) {
            Some((name, ep)) => {
                // The accelerator does the work; a thread or two feeds it.
                let built = Session::builder(&l.env)
                    .and_then(|b| b.with_intra_threads(threads.clamp(1, 2) as usize).map_err(Into::into))
                    .and_then(|b| b.with_memory_pattern(false).map_err(Into::into))
                    .and_then(|b| b.with_execution_providers([ep]).map_err(Into::into))
                    .and_then(|mut b| b.commit_from_file(path));
                match built {
                    Ok(s) => {
                        record(path, name);
                        return Ok(s);
                    }
                    Err(e) if want == Compute::Gpu => return Err(fail(e)),
                    Err(e) => eprintln!("lana: {} stays on the CPU: {name} refused it: {e}", path.display()),
                }
            }
            None if want == Compute::Gpu => return Err("this computer has no accelerator Lana can use".into()),
            None => {}
        }
    }
    let s = cpu_session(path, threads)?;
    record(path, "CPU");
    Ok(s)
}

/// A model's session, on the accelerator or the CPU, that falls back to the
/// CPU if the accelerator fails it while it runs: a model can load on a GPU
/// and still fail there on its first input (cuDNN without kernels for an
/// older card, an operator CoreML compiled but cannot execute).
pub struct Model {
    session: Session,
    path: PathBuf,
    threads: i32,
    accelerated: bool,
}

impl Model {
    pub fn load(path: &Path, role: Role, threads: i32) -> Result<Self, String> {
        let session = session(path, role, threads)?;
        let accelerated = STATE.lock().unwrap().used.get(&key(path)).is_some_and(|on| *on != "CPU");
        Ok(Model { session, path: path.to_path_buf(), threads, accelerated })
    }

    /// Run `f` on the session. `f` builds the inputs itself and copies out
    /// what it needs, so it can be run again: on an accelerator error the
    /// model is loaded on the CPU and `f` runs once more there. With
    /// compute = gpu the error stands.
    pub fn with<R>(&mut self, mut f: impl FnMut(&mut Session) -> Result<R, String>) -> Result<R, String> {
        match f(&mut self.session) {
            Err(e) if self.accelerated && compute() != Compute::Gpu => {
                eprintln!("lana: {} failed on the accelerator, now on the CPU: {e}", self.path.display());
                self.session = cpu_session(&self.path, self.threads)?;
                self.accelerated = false;
                record(&self.path, "CPU (the accelerator failed it)");
                f(&mut self.session)
            }
            r => r,
        }
    }
}

/// A CPU session. Optimising a model's graph is most of what loading it
/// costs (ZIPA small: 1.34 s, of which 0.38 s is left when the optimised
/// graph is read back), and Lana loads a model for every job, so the
/// optimised graph is kept in the models folder and read back next time.
/// Only for the CPU: an accelerator's optimisations may not survive being
/// saved. Not for models whose weights sit in files beside them (MMS):
/// each saved copy would hold all of them. Not for models over
/// SAVE_LIMIT either, where reading dominates: Parakeet's 650 MB encoder
/// loads in 1.44 s instead of 2.22 s, for another 650 MB on disk.
fn cpu_session(path: &Path, threads: i32) -> Result<Session, String> {
    let l = loaded()?;
    let build = || Session::builder(&l.env).and_then(|b| b.with_intra_threads(threads.max(1) as usize).map_err(Into::into));
    let plain = || build().and_then(|mut b| b.commit_from_file(path)).map_err(|e| format!("could not load {}: {e}", path.display()));
    let Some(saved) = optimised_copy(path) else { return plain() };
    if saved.is_file() {
        use ort::session::builder::GraphOptimizationLevel;
        match build().and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Disable).map_err(Into::into))
            .and_then(|mut b| b.commit_from_file(&saved))
        {
            Ok(s) => {
                // Its date says when it was last used, for `prune`.
                let _ = std::fs::File::options().append(true).open(&saved)
                    .and_then(|f| f.set_modified(std::time::SystemTime::now()));
                return Ok(s);
            }
            Err(e) => {
                eprintln!("lana: the saved optimised graph of {} did not load, so it is made again: {e}", path.display());
                let _ = std::fs::remove_file(&saved);
            }
        }
    }
    let _ = std::fs::create_dir_all(saved.parent().unwrap_or(Path::new(".")));
    let mut tmp = saved.clone().into_os_string();
    tmp.push(".part");
    let tmp = PathBuf::from(tmp);
    match build().and_then(|b| b.with_optimized_model_path(&tmp).map_err(Into::into)).and_then(|mut b| b.commit_from_file(path)) {
        Ok(s) => {
            let _ = std::fs::rename(&tmp, &saved);
            Ok(s)
        }
        Err(_) => {
            let _ = std::fs::remove_file(&tmp);
            plain()
        }
    }
}

/// Delete optimised graphs unused for a month: those of deleted models, and
/// of an older runtime or model version.
pub fn prune() {
    let Some(dir) = STATE.lock().unwrap().cache.clone().map(|d| d.join("optimised")) else { return };
    let month = std::time::Duration::from_secs(30 * 24 * 3600);
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let old = e.metadata().and_then(|m| m.modified()).ok()
            .and_then(|t| t.elapsed().ok()).is_some_and(|age| age > month);
        if old {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

/// The largest model whose optimised graph is kept (see cpu_session).
const SAVE_LIMIT: u64 = 400_000_000;

/// Where a model's optimised graph is kept: in the cache folder, named by
/// the model's path, size and date and the runtime's version, so a changed
/// model or runtime makes a new one. None if there is nowhere to keep it or
/// the model keeps weights beside it.
fn optimised_copy(path: &Path) -> Option<PathBuf> {
    use sha2::{Digest, Sha256};
    let dir = STATE.lock().unwrap().cache.clone()?;
    let meta = std::fs::metadata(path).ok().filter(|m| m.len() <= SAVE_LIMIT)?;
    let parent = path.parent()?;
    let beside = std::fs::read_dir(parent).ok()?.flatten().any(|e| {
        let n = e.file_name().to_string_lossy().to_lowercase();
        n.ends_with(".bin") || n.ends_with(".data") || n.ends_with(".weights")
    });
    if beside {
        return None;
    }
    let mut h = Sha256::new();
    h.update(path.to_string_lossy().as_bytes());
    h.update(meta.len().to_le_bytes());
    if let Ok(t) = meta.modified().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)) {
        h.update(t.as_secs().to_le_bytes());
    }
    h.update(ort::info().as_bytes());
    let hex: String = h.finalize().iter().take(12).map(|b| format!("{b:02x}")).collect();
    Some(dir.join("optimised").join(format!("{}-{hex}.onnx", path.file_stem()?.to_string_lossy())))
}

fn key(path: &Path) -> String {
    path.iter().rev().take(2).collect::<Vec<_>>().into_iter().rev()
        .map(|p| p.to_string_lossy()).collect::<Vec<_>>().join("/")
}

fn record(path: &Path, on: &'static str) {
    STATE.lock().unwrap().used.insert(key(path), on);
}

/// What the runtime is and where each model ran, for the settings screen,
/// bug reports and lana-bench.
pub fn describe() -> Value {
    let s = STATE.lock().unwrap();
    let used: serde_json::Map<String, Value> = s.used.iter().map(|(k, v)| (k.clone(), json!(v))).collect();
    let compute = s.compute.name();
    drop(s);
    let (library, accelerator, error) = match loaded() {
        Ok(l) => (Some(l.library.display().to_string()), accelerator(l).map(|a| a.0), None),
        Err(e) => (None, None, Some(e)),
    };
    json!({ "compute": compute, "library": library, "accelerator": accelerator, "error": error, "models": used })
}
