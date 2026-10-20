// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "LanaOmnilingual",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/soniqo/speech-swift.git",
                 revision: "3f58a58a8f867a4c87b97dbf4d6050d6a88ef665"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.4"),
    ],
    targets: [.executableTarget(name: "lana-omnilingual", dependencies: [
        .product(name: "OmnilingualASR", package: "speech-swift"),
        .product(name: "AudioCommon", package: "speech-swift"),
        .product(name: "MLX", package: "mlx-swift"),
    ], path: "Sources")],
    swiftLanguageModes: [.v5]
)
