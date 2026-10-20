import Foundation
import OmnilingualASR
import AudioCommon
import MLX
import Darwin

@main struct LanaOmnilingual {
    static func main() async {
        do {
            if CommandLine.arguments.count == 2 && ["--self-test", "--self-test-metal"].contains(CommandLine.arguments[1]) {
                // Hosted build runners may lack a Metal device. A real Mac
                // can additionally exercise the bundled shaders explicitly.
                let device: Device = CommandLine.arguments[1] == "--self-test-metal" ? .gpu : .cpu
                let result = Device.withDefaultDevice(device) {
                    let x = MLXArray([Float(1), Float(2)])
                    let sum = (x * x).sum()
                    eval(sum)
                    return sum.item(Float.self)
                }
                guard result == 5 else { throw WorkerError.message("MLX self-test failed") }
                try send(["protocol": "lana-mac-1", "self_test": true])
                return
            }
            guard CommandLine.arguments.count == 3 else { throw WorkerError.message("expected model directory and variant") }
            let directory = URL(fileURLWithPath: CommandLine.arguments[1])
            let variant = CommandLine.arguments[2] == "1B" ? OmnilingualMLXConfig.Variant.b1 : .m300
            let model = try await OmnilingualASRMLXModel.fromPretrained(
                variant: variant, bits: 8, cacheDir: directory, offlineMode: true)
            let vocabulary = try SentencePieceModel(contentsOf: directory.appendingPathComponent("tokenizer.model"))
            // Lana's CTC decoder expects blank at 0; Meta uses pad at 1.
            var tokens = vocabulary.pieces.map { $0.isControlOrUnknown ? "" : $0.text }
            tokens.swapAt(0, 1)
            tokens[0] = ""
            tokens[1] = ""
            try send(["protocol": "lana-mac-1", "backend": "MLX (Metal, 8-bit)", "tokens": tokens])
            while let samples = try readSamples() {
                // Lana supplies the same utterance-normalized waveform as ONNX.
                let input = MLXArray(samples).reshaped([1, samples.count, 1])
                let logits = model.ctcHead(model.encoder(model.frontend(input)))
                eval(logits)
                let shape = logits.shape
                guard shape.count == 3, shape[0] == 1, shape[2] == tokens.count else {
                    throw WorkerError.message("unexpected Omnilingual logits shape \(shape)")
                }
                var values = logits.asType(.float32).asArray(Float.self)
                for row in 0..<shape[1] { values.swapAt(row * shape[2], row * shape[2] + 1) }
                try send(["protocol": "lana-mac-1", "t": shape[1], "v": shape[2], "peak_bytes": peakBytes()])
                let data = values.withUnsafeBytes { Data($0) }
                try FileHandle.standardOutput.write(contentsOf: data)
            }
        } catch {
            FileHandle.standardError.write(Data("lana-omnilingual: \(error)\n".utf8))
            exit(1)
        }
    }
}
