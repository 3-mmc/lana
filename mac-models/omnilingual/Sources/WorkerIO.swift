import Foundation
import Darwin

enum WorkerError: Error { case message(String) }

func send(_ value: [String: Any]) throws {
    var data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    data.append(10)
    try FileHandle.standardOutput.write(contentsOf: data)
}

func readExactly(_ count: Int) throws -> Data {
    var data = Data()
    while data.count < count {
        guard let part = try FileHandle.standardInput.read(upToCount: count - data.count), !part.isEmpty else {
            throw WorkerError.message("incomplete audio request")
        }
        data.append(part)
    }
    return data
}

func readSamples() throws -> [Float]? {
    guard let first = try FileHandle.standardInput.read(upToCount: 1), !first.isEmpty else { return nil }
    var header = first
    header.append(try readExactly(7))
    let count = header.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
    guard count >= 400, count <= 640_000 else { throw WorkerError.message("audio must be between 25 ms and 40 s at 16 kHz") }
    let data = try readExactly(Int(count) * 4)
    return data.withUnsafeBytes { raw in
        (0..<Int(count)).map { Float(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self).littleEndian) }
    }
}

func peakBytes() -> UInt64 {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return UInt64(max(0, usage.ru_maxrss))
}
