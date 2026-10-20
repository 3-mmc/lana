// SPDX-License-Identifier: GPL-3.0-or-later
// Native Mac assets, using the same resumable, checksum-verified downloader.
use super::models::{File, Variant};

pub const OMNI_300: Variant = Variant { precision: "MLX 8-bit", size_mb: 360, files: &[
    File { repo: "aufklarer/Omnilingual-ASR-CTC-300M-MLX-8bit", path: "model.safetensors", local: "mac-mlx-300m/model.safetensors" },
    File { repo: "aufklarer/Omnilingual-ASR-CTC-300M-MLX-8bit", path: "config.json", local: "mac-mlx-300m/config.json" },
    File { repo: "aufklarer/Omnilingual-ASR-CTC-300M-MLX-8bit", path: "tokenizer.model", local: "mac-mlx-300m/tokenizer.model" },
] };

pub const OMNI_1B: Variant = Variant { precision: "MLX 8-bit", size_mb: 1055, files: &[
    File { repo: "aufklarer/Omnilingual-ASR-CTC-1B-MLX-8bit", path: "model.safetensors", local: "mac-mlx-1b/model.safetensors" },
    File { repo: "aufklarer/Omnilingual-ASR-CTC-1B-MLX-8bit", path: "config.json", local: "mac-mlx-1b/config.json" },
    File { repo: "aufklarer/Omnilingual-ASR-CTC-1B-MLX-8bit", path: "tokenizer.model", local: "mac-mlx-1b/tokenizer.model" },
] };

pub const PARAKEET: Variant = Variant { precision: "CoreML int8", size_mb: 484, files: &[
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "parakeet_vocab.json", local: "mac-coreml/parakeet_vocab.json" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Decoder.mlmodelc/analytics/coremldata.bin", local: "mac-coreml/Decoder.mlmodelc/analytics/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Decoder.mlmodelc/coremldata.bin", local: "mac-coreml/Decoder.mlmodelc/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Decoder.mlmodelc/metadata.json", local: "mac-coreml/Decoder.mlmodelc/metadata.json" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Decoder.mlmodelc/model.mil", local: "mac-coreml/Decoder.mlmodelc/model.mil" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Decoder.mlmodelc/weights/weight.bin", local: "mac-coreml/Decoder.mlmodelc/weights/weight.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Encoder.mlmodelc/analytics/coremldata.bin", local: "mac-coreml/Encoder.mlmodelc/analytics/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Encoder.mlmodelc/coremldata.bin", local: "mac-coreml/Encoder.mlmodelc/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Encoder.mlmodelc/metadata.json", local: "mac-coreml/Encoder.mlmodelc/metadata.json" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Encoder.mlmodelc/model.mil", local: "mac-coreml/Encoder.mlmodelc/model.mil" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Encoder.mlmodelc/weights/weight.bin", local: "mac-coreml/Encoder.mlmodelc/weights/weight.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "JointDecisionv3.mlmodelc/analytics/coremldata.bin", local: "mac-coreml/JointDecisionv3.mlmodelc/analytics/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "JointDecisionv3.mlmodelc/coremldata.bin", local: "mac-coreml/JointDecisionv3.mlmodelc/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "JointDecisionv3.mlmodelc/metadata.json", local: "mac-coreml/JointDecisionv3.mlmodelc/metadata.json" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "JointDecisionv3.mlmodelc/model.mil", local: "mac-coreml/JointDecisionv3.mlmodelc/model.mil" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "JointDecisionv3.mlmodelc/weights/weight.bin", local: "mac-coreml/JointDecisionv3.mlmodelc/weights/weight.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Preprocessor.mlmodelc/analytics/coremldata.bin", local: "mac-coreml/Preprocessor.mlmodelc/analytics/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Preprocessor.mlmodelc/coremldata.bin", local: "mac-coreml/Preprocessor.mlmodelc/coremldata.bin" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Preprocessor.mlmodelc/metadata.json", local: "mac-coreml/Preprocessor.mlmodelc/metadata.json" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Preprocessor.mlmodelc/model.mil", local: "mac-coreml/Preprocessor.mlmodelc/model.mil" },
    File { repo: "FluidInference/parakeet-tdt-0.6b-v3-coreml", path: "Preprocessor.mlmodelc/weights/weight.bin", local: "mac-coreml/Preprocessor.mlmodelc/weights/weight.bin" },
] };

pub fn variant(id: &str) -> Option<&'static Variant> {
    match id {
        "omnilingual-300m" => Some(&OMNI_300),
        "omnilingual-1b" => Some(&OMNI_1B),
        "parakeet-tdt-0.6b-v3" => Some(&PARAKEET),
        _ => None,
    }
}
