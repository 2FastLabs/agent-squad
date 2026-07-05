import AVFoundation
import os

/// Played-milliseconds accounting for `AudioOutput.playedMilliseconds()` — feeds the session's
/// `conversation.item.truncate` on barge-in. Buffers report back through the player's
/// `.dataPlayedBack` completion callbacks; a *burst* is one continuous run of playback starting
/// on an empty queue (in practice: one spoken response).
///
/// `flush()` (barge-in) voids the pending buffers' callbacks — `AVAudioPlayerNode.stop()` fires
/// them for *unplayed* buffers too — but KEEPS the played total: the session samples it right
/// after the cut. The total resets when the next burst begins.
///
/// If the queue drains *mid-response* (a stall — rare, since the server sends audio faster than
/// realtime), the resume starts a new burst and the total restarts. Deliberately conservative:
/// truncation then under-reports played audio, never claiming the user heard something they
/// didn't (`audio_end_ms` beyond the played point is the harmful direction).
///
/// Sendable via `OSAllocatedUnfairLock`; completion callbacks arrive on AVFoundation's internal
/// queue and never touch the owning class's state, so no lock-ordering hazard with `stop()`.
final class PlaybackClock: Sendable {
    private struct State {
        var playedMs = 0.0
        var pending = 0
        var generation = 0
        var idle = true   // queue ran empty → the next schedule starts a new burst
    }
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// Call before scheduling a buffer; pass the returned token to `completed`.
    func willSchedule() -> Int {
        state.withLock {
            if $0.idle { $0.playedMs = 0; $0.idle = false }
            $0.pending += 1
            return $0.generation
        }
    }

    /// From the buffer's `.dataPlayedBack` completion callback.
    func completed(durationMs: Double, token: Int) {
        state.withLock {
            guard token == $0.generation else { return }   // voided by flush — stop() fires these early
            $0.playedMs += durationMs
            $0.pending -= 1
            if $0.pending == 0 { $0.idle = true }
        }
    }

    /// Barge-in cut: void pending callbacks, keep the played total for sampling.
    func flushed() {
        state.withLock {
            $0.generation += 1
            $0.pending = 0
            $0.idle = true
        }
    }

    func reset() {
        // Bump the generation, don't rewind it — `player.stop()` fires pending callbacks
        // asynchronously and a rewound generation would let them pass the token guard.
        state.withLock { $0 = State(generation: $0.generation + 1) }
    }

    func milliseconds() -> Double {
        state.withLock { $0.playedMs }
    }
}
