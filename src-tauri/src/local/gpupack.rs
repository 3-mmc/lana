// SPDX-License-Identifier: GPL-3.0-or-later
// The GPU pack: on Linux with an NVIDIA card, the CUDA build of ONNX Runtime
// and the CUDA and cuDNN libraries it needs, downloaded on request into the
// models folder (runtime::GPU_PACK), where runtime.rs prefers it to the CPU
// build Lana ships. About 1.8 GB to download and 2.6 GB on disk, so it is
// never part of the app itself.
//
// Everything comes from where its makers publish it: ONNX Runtime from
// Microsoft's GitHub releases, CUDA and cuDNN as NVIDIA's own wheels on PyPI
// (under NVIDIA's licence; Lana fetches them for the user and hosts nothing).
// Each file is pinned by SHA-256, and only the libraries are kept.
//
// CUDA 12 rather than 13, and cuDNN 9.10.2 rather than the newest: CUDA 13
// and cuDNN 9.11 dropped the GTX 10xx cards (Pascal, compute capability
// 6.x). Measured on a GTX 1080 on 2026-10-19: cuDNN 9.11 and 9.27 fail with
// "no kernel image is available for execution on the device", 9.10.2 runs.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::{models::Models, runtime};

/// The progress entry downloads are reported under, beside the models'.
pub const ID: &str = "gpu-pack";

struct Package {
    url: &'static str,
    sha256: &'static str,
    size: u64,
    /// What to keep: archive members whose file name is listed (or starts
    /// with a listed prefix ending in '*'), saved under that name; the first
    /// pair renames.
    keep: &'static [(&'static str, &'static str)],
}

const PACKAGES: &[Package] = &[
    Package {
        url: "https://github.com/microsoft/onnxruntime/releases/download/v1.30.0/onnxruntime-linux-x64-gpu_cuda12-1.30.0.tgz",
        sha256: "f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1",
        size: 439_354_926,
        keep: &[("libonnxruntime.so.1.30.0", "libonnxruntime.so"),
                ("libonnxruntime_providers_cuda.so", "libonnxruntime_providers_cuda.so"),
                ("libonnxruntime_providers_shared.so", "libonnxruntime_providers_shared.so"),
                ("LICENSE", "onnxruntime-LICENSE"), ("ThirdPartyNotices.txt", "onnxruntime-ThirdPartyNotices.txt")],
    },
    Package {
        url: "https://files.pythonhosted.org/packages/bc/46/a92db19b8309581092a3add7e6fceb4c301a3fd233969856a8cbf042cd3c/nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3",
        size: 3_493_179,
        keep: &[("libcudart.so.12", "libcudart.so.12"), ("License.txt", "NVIDIA-License.txt")],
    },
    Package {
        url: "https://files.pythonhosted.org/packages/cb/c0/0a517bfe63ccd3b92eb254d264e28fca3c7cab75d07daea315250fb1bf73/nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702",
        size: 581_240_110,
        keep: &[("libcublas.so.12", "libcublas.so.12"), ("libcublasLt.so.12", "libcublasLt.so.12")],
    },
    Package {
        url: "https://files.pythonhosted.org/packages/31/44/193a0e171750ca9f8320626e8a1f2381e4077a65e69e2fb9708bd479e34a/nvidia_curand_cu12-10.3.10.19-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "49b274db4780d421bd2ccd362e1415c13887c53c214f0d4b761752b8f9f6aa1e",
        size: 68_295_626,
        keep: &[("libcurand.so.10", "libcurand.so.10")],
    },
    Package {
        url: "https://files.pythonhosted.org/packages/ba/51/e123d997aa098c61d029f76663dedbfb9bc8dcf8c60cbd6adbe42f76d049/nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "949452be657fa16687d0930933f032835951ef0892b37d2d53824d1a84dc97a8",
        size: 706_758_467,
        keep: &[("libcudnn*", "")],
    },
];

/// Whether this computer could use the pack: Linux with NVIDIA's driver.
pub fn wanted() -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64"))
        && (Path::new("/proc/driver/nvidia/version").exists() || Path::new("/usr/lib/wsl/lib/libcuda.so.1").exists())
}

fn dir(m: &Models) -> PathBuf {
    m.dir.join(runtime::GPU_PACK)
}

pub fn installed(m: &Models) -> bool {
    dir(m).join("libonnxruntime.so").is_file()
}

/// For the settings screen.
pub fn status(m: &Models) -> Value {
    let d = m.downloads.lock().unwrap().get(ID).cloned();
    json!({
        "wanted": wanted(),
        "installed": installed(m),
        "size_mb": PACKAGES.iter().map(|p| p.size).sum::<u64>() / 1_000_000,
        "downloading": d.as_ref().filter(|d| d.error.is_none()).map(|d| json!({ "done": d.done, "total": d.total })),
        "error": d.and_then(|d| d.error),
    })
}

pub fn delete(m: &Models) -> Result<(), String> {
    for d in [dir(m), part(m)] {
        match std::fs::remove_dir_all(&d) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.to_string()),
            _ => {}
        }
    }
    Ok(())
}

/// The pack is assembled here and renamed into place when complete, so a
/// half-downloaded pack is never loaded.
fn part(m: &Models) -> PathBuf {
    let mut p = dir(m).into_os_string();
    p.push(".part");
    PathBuf::from(p)
}

/// Start the download in the background; poll `status`.
pub fn start(m: &Models) -> Result<(), String> {
    if !wanted() {
        return Err("The GPU pack is for Linux computers with an NVIDIA graphics card.".into());
    }
    {
        let mut dl = m.downloads.lock().unwrap();
        if dl.get(ID).is_some_and(|d| d.error.is_none()) {
            return Ok(());
        }
        dl.insert(ID.to_owned(), Default::default());
    }
    let m = m.clone();
    tokio::spawn(async move {
        let result = fetch(&m).await;
        let mut dl = m.downloads.lock().unwrap();
        match result {
            Ok(()) => {
                dl.remove(ID);
            }
            Err(msg) => {
                dl.insert(ID.to_owned(), super::models::Download { error: Some(msg), ..Default::default() });
            }
        }
    });
    Ok(())
}

async fn fetch(m: &Models) -> Result<(), String> {
    let part = part(m);
    let archives = part.join("downloads");
    tokio::fs::create_dir_all(&archives).await.map_err(|e| e.to_string())?;
    let total: u64 = PACKAGES.iter().map(|p| p.size).sum();
    let mut before = 0;
    for p in PACKAGES {
        let name = p.url.rsplit('/').next().unwrap_or("package");
        let archive = archives.join(name);
        if !archive.is_file() {
            m.fetch_file(ID, p.url, &archive, p.sha256, p.size, before, total).await?;
        }
        before += p.size;
        let (archive2, part2) = (archive.clone(), part.clone());
        tokio::task::spawn_blocking(move || unpack(&archive2, &part2, p.keep))
            .await
            .map_err(|e| e.to_string())??;
        let _ = tokio::fs::remove_file(&archive).await;
    }
    let _ = tokio::fs::remove_dir(&archives).await;
    let done = dir(m);
    let _ = tokio::fs::remove_dir_all(&done).await;
    tokio::fs::rename(&part, &done).await.map_err(|e| e.to_string())?;
    Ok(())
}

/// The name a member is kept under, if it is kept.
fn kept(member: &str, keep: &[(&str, &str)]) -> Option<String> {
    let base = member.rsplit('/').next()?;
    keep.iter().find_map(|(want, save)| match want.strip_suffix('*') {
        Some(prefix) => base.starts_with(prefix).then(|| base.to_owned()),
        None => (base == *want).then(|| save.to_string()),
    })
}

/// Take the kept members out of a .tgz or a wheel (a zip) into `to`.
fn unpack(archive: &Path, to: &Path, keep: &[(&str, &str)]) -> Result<(), String> {
    let fail = |e: &dyn std::fmt::Display| format!("Could not unpack {}: {e}", archive.display());
    let file = std::fs::File::open(archive).map_err(|e| fail(&e))?;
    if archive.extension().is_some_and(|e| e == "whl") {
        let mut zip = zip::ZipArchive::new(file).map_err(|e| fail(&e))?;
        for i in 0..zip.len() {
            let mut entry = zip.by_index(i).map_err(|e| fail(&e))?;
            if let Some(name) = kept(entry.name(), keep) {
                let mut out = std::fs::File::create(to.join(name)).map_err(|e| fail(&e))?;
                std::io::copy(&mut entry, &mut out).map_err(|e| fail(&e))?;
            }
        }
    } else {
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        for entry in tar.entries().map_err(|e| fail(&e))? {
            let mut entry = entry.map_err(|e| fail(&e))?;
            if !entry.header().entry_type().is_file() {
                continue;
            }
            let path = entry.path().map_err(|e| fail(&e))?.to_string_lossy().into_owned();
            if let Some(name) = kept(&path, keep) {
                let mut out = std::fs::File::create(to.join(name)).map_err(|e| fail(&e))?;
                std::io::copy(&mut entry, &mut out).map_err(|e| fail(&e))?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn members_are_kept_by_name_or_prefix() {
        let keep = PACKAGES[0].keep;
        assert_eq!(kept("onnxruntime-linux-x64-gpu_cuda12-1.30.0/lib/libonnxruntime.so.1.30.0", keep).as_deref(), Some("libonnxruntime.so"));
        assert_eq!(kept("onnxruntime-linux-x64-gpu_cuda12-1.30.0/lib/libonnxruntime_providers_tensorrt.so", keep), None);
        let cudnn = PACKAGES[4].keep;
        assert_eq!(kept("nvidia/cudnn/lib/libcudnn_ops.so.9", cudnn).as_deref(), Some("libcudnn_ops.so.9"));
        assert_eq!(kept("nvidia/cudnn/include/cudnn.h", cudnn), None);
    }
}
