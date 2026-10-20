import Foundation
import FluidAudio
import Darwin

// All model files are downloaded and verified by Lana. Never download here.
@main struct LanaParakeet {
    static func main() async {
        do {
            if CommandLine.arguments.count == 2 && CommandLine.arguments[1] == "--self-test" {
                try send(["protocol": "lana-mac-1", "self_test": true])
                return
            }
            guard CommandLine.arguments.count == 2 else { throw WorkerError.message("expected model directory") }
            let directory = URL(fileURLWithPath: CommandLine.arguments[1])
            let models = try AsrModels.loadLocal(from: directory, version: .v3)
            let manager = AsrManager(config: .default)
            try await manager.loadModels(models)
            try send(["protocol": "lana-mac-1", "backend": "CoreML (FluidAudio, CPU and Neural Engine)"])
            while let samples = try readSamples() {
                var state = try TdtDecoderState()
                let result = try await manager.transcribe(samples, decoderState: &state)
                let words = buildWordTimings(from: result.tokenTimings ?? []).map {
                    ["text": $0.word, "start": $0.startTime, "end": $0.endTime] as [String: Any]
                }
                guard result.text.isEmpty || !words.isEmpty else { throw WorkerError.message("Parakeet returned text without word timestamps") }
                try send(["protocol": "lana-mac-1", "words": words, "peak_bytes": peakBytes()])
            }
        } catch {
            FileHandle.standardError.write(Data("lana-parakeet: \(error)\n".utf8))
            exit(1)
        }
    }
}
