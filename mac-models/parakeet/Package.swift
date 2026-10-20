// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LanaParakeet",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git",
                 revision: "61d41b3ca04833d1cb80632c82ba401f117a01ef"),
    ],
    targets: [.executableTarget(name: "lana-parakeet", dependencies: [
        .product(name: "FluidAudio", package: "FluidAudio"),
    ], path: "Sources")],
    swiftLanguageModes: [.v5]
)
