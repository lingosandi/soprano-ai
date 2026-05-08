/**
 * AudioStreamPlayer — low-latency PCM audio playback via the Web Audio API.
 *
 * Queues incoming Float32 PCM chunks and plays them back-to-back with
 * minimal gap. Uses AudioContext + createBufferSource for each chunk so
 * samples start playing the moment they are decoded.
 */

import { PLAYBACK_SAMPLE_RATE } from "./config"
import type { IAudioPlayer } from "./voice-agent-types"

const SAMPLE_RATE = PLAYBACK_SAMPLE_RATE

export class AudioStreamPlayer implements IAudioPlayer {
    private ctx: AudioContext | null = null
    private nextStartTime = 0
    private gainNode: GainNode | null = null
    private analyser: AnalyserNode | null = null
    private analyserData: Uint8Array<ArrayBuffer> | null = null

    /** Initialize the AudioContext. Must be called from a user gesture. */
    async init(): Promise<void> {
        if (this.ctx && this.ctx.state !== "closed") {
            // Resume if suspended (e.g. after browser auto-suspend policy)
            if (this.ctx.state === "suspended") await this.ctx.resume()
            return
        }
        this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
        this.gainNode = this.ctx.createGain()

        // Analyser for output level metering
        this.analyser = this.ctx.createAnalyser()
        this.analyser.fftSize = 256
        this.analyser.smoothingTimeConstant = 0.8

        // Route: gainNode → analyser → destination
        this.gainNode.connect(this.analyser)
        this.analyser.connect(this.ctx.destination)
        this.analyserData = new Uint8Array(this.analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>

        this.nextStartTime = 0
        // Ensure playback is allowed (some webviews start suspended)
        if (this.ctx.state === "suspended") await this.ctx.resume()
    }

    /** Enqueue Float32 PCM samples for immediate (gapless) playback. */
    enqueue(samples: Float32Array): void {
        if (!this.ctx || !this.gainNode) return
        if (samples.length === 0) return

        // WebView2 (and some browsers) can auto-suspend the AudioContext
        // after a period of silence/inactivity.  Scheduled buffer sources
        // still "run" in a suspended context (currentTime may advance) but
        // produce no audible output — the user sees "speaking" with silence.
        // Resume eagerly so audio is never silently dropped.
        if (this.ctx.state === "suspended") {
            console.warn("[AudioStreamPlayer] AudioContext was suspended — resuming before enqueue")
            this.ctx.resume().catch(() => {})
        }

        const buffer = this.ctx.createBuffer(1, samples.length, SAMPLE_RATE)
        buffer.getChannelData(0).set(samples)

        const source = this.ctx.createBufferSource()
        source.buffer = buffer
        source.connect(this.gainNode)

        // Schedule right after the last queued chunk (or now)
        const now = this.ctx.currentTime
        if (this.nextStartTime < now) {
            this.nextStartTime = now
        }
        source.start(this.nextStartTime)
        this.nextStartTime += buffer.duration
    }

    /** Interrupt playback by stopping & resetting. */
    stop(): void {
        if (this.ctx) {
            // Disconnect everything by closing and recreating
            this.ctx.close().catch(() => {})
            this.ctx = null
            this.gainNode = null
            this.analyser = null
            this.analyserData = null
        }
        this.nextStartTime = 0
    }

    /** Get the current output audio level (0–1). */
    getAudioLevel(): number {
        if (!this.analyser || !this.analyserData) return 0
        this.analyser.getByteFrequencyData(this.analyserData)
        let sum = 0
        for (const v of this.analyserData) sum += v
        return sum / (this.analyserData.length * 255)
    }

    /** True when audio has been queued and hasn't finished yet. */
    get isPlaying(): boolean {
        if (!this.ctx) return false
        return this.ctx.currentTime < this.nextStartTime
    }

    /**
     * Returns a promise that resolves when all queued audio has finished playing.
     *
     * Uses a settle-check: requires two consecutive polls where isPlaying is
     * false before resolving.  This prevents premature resolution when there
     * is a brief gap between enqueued chunks (e.g. Cartesia sends the last
     * few audio chunks close together and `ctx.currentTime` momentarily
     * catches up to `nextStartTime` between them).
     */
    waitForEnd(): Promise<void> {
        return new Promise((resolve) => {
            let consecutiveIdle = 0
            const REQUIRED_IDLE_CHECKS = 2
            const check = () => {
                if (!this.isPlaying) {
                    consecutiveIdle++
                    if (consecutiveIdle >= REQUIRED_IDLE_CHECKS) {
                        resolve()
                        return
                    }
                } else {
                    consecutiveIdle = 0
                }
                setTimeout(check, 100)
            }
            check()
        })
    }
}
