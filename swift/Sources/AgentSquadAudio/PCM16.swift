import AVFoundation

/// Shared PCM16 ↔ engine-format conversion used by `MicCapture`, `AudioPlayback`, and
/// `VoiceProcessedAudioIO` — one copy of the subtle real-time code.
enum PCM16 {
    /// Runs on the real-time audio thread: convert one captured buffer to PCM16 at
    /// `targetFormat`'s rate and yield it into the stream.
    static func convertAndYield(
        _ buffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        targetFormat: AVAudioFormat,
        continuation: AsyncStream<Data>.Continuation
    ) {
        let ratio = targetFormat.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 1_024
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        // The block form is REQUIRED for sample-rate conversion (48k→24k) — `convert(to:from:)`
        // throws on differing sample rates. Feeding one input buffer per call with the converter
        // reused across taps is the correct *streaming* idiom: it carries resampler filter state
        // between calls for continuity. The input block is `@Sendable` but called synchronously on
        // this thread, so single-threaded access to these captures is safe.
        nonisolated(unsafe) let source = buffer
        nonisolated(unsafe) var provided = false
        var conversionError: NSError?
        converter.convert(to: output, error: &conversionError) { _, status in
            if provided { status.pointee = .noDataNow; return nil }
            provided = true
            status.pointee = .haveData
            return source
        }
        guard conversionError == nil, output.frameLength > 0, let samples = output.int16ChannelData else { return }
        continuation.yield(Data(bytes: samples[0], count: Int(output.frameLength) * MemoryLayout<Int16>.size))
    }

    /// PCM16 LE mono bytes → float32 buffer ready to schedule on a player node.
    static func floatBuffer(fromPCM16 data: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let samples = AudioPlayback.floatSamples(fromPCM16: data)
        guard !samples.isEmpty, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return nil }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channel = buffer.floatChannelData![0]
        for i in samples.indices { channel[i] = samples[i] }
        return buffer
    }
}
