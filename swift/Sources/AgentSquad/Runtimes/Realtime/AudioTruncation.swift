import Foundation

/// Per-turn bookkeeping for barge-in truncation (`conversation.item.truncate`): which assistant
/// audio item is playing and how many milliseconds of it the server has sent. Compared against the
/// playback clock's *played* milliseconds to build a truncate frame that never exceeds the item's
/// actual duration (the API errors on `audio_end_ms` greater than the audio's length).
struct AudioTruncationState {
    private(set) var itemId: String?
    private(set) var receivedMs: Double = 0
    // Audio played before this item within the same playback burst (a tool→continue turn can
    // speak several items back-to-back without the queue draining) — subtracted from the burst
    // clock so `audio_end_ms` is per-item, never inflated by a predecessor's playback.
    private(set) var itemStartMs: Double = 0

    /// Record one relayed audio delta. A new `itemId` restarts the received counter and moves
    /// the burst offset past the previous item's audio.
    mutating func record(itemId: String, pcm16ByteCount: Int, sampleRate: Int) {
        guard !itemId.isEmpty, sampleRate > 0 else { return }
        if itemId != self.itemId {
            itemStartMs += receivedMs
            self.itemId = itemId
            receivedMs = 0
        }
        // PCM16 mono: 2 bytes per sample.
        receivedMs += Double(pcm16ByteCount) / 2 / Double(sampleRate) * 1_000
    }

    mutating func reset() {
        itemId = nil
        receivedMs = 0
        itemStartMs = 0
    }

    /// The truncate frame for a barge-in, or `nil` when there is nothing to truncate: no item
    /// relayed, no playback measurement, or nothing of THIS item unplayed. `played` can go
    /// negative when the burst drained between items (the clock restarted) and can exceed
    /// `receivedMs` on a stale clock — both skip rather than send a value the server could
    /// reject (`audio_end_ms` beyond the actual duration errors) or that over-reports.
    func truncateFrame(playedMs: Double?) -> String? {
        guard let itemId, let playedMs else { return nil }
        let played = playedMs - itemStartMs
        guard played >= 0, played < receivedMs else { return nil }
        return RealtimeWire.truncateItem(itemId: itemId, audioEndMs: Int(played))
    }
}
