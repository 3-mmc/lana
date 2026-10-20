// whisper.cpp's Metal code uses @available, which calls Clang's platform
// version helpers. Rust links with -nodefaultlibs, so add the matching Clang
// runtime explicitly for both the app and lana-bench.
pub fn configure() {
    println!("cargo:rerun-if-changed=../scripts/macos-link.rs");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    let output = std::process::Command::new("xcrun")
        .args(["--sdk", "macosx", "clang", "--print-resource-dir"])
        .output()
        .expect("finding the Xcode Clang runtime requires xcrun");
    assert!(
        output.status.success(),
        "xcrun could not find the Clang runtime: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let resource_dir = String::from_utf8(output.stdout).expect("Clang resource path is not UTF-8");
    let library_dir = std::path::Path::new(resource_dir.trim()).join("lib/darwin");
    assert!(
        library_dir.join("libclang_rt.osx.a").is_file(),
        "Xcode's macOS Clang runtime is missing from {}",
        library_dir.display()
    );
    println!("cargo:rustc-link-search=native={}", library_dir.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}
