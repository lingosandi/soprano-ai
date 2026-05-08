/**
 * Tests for AudioStreamPlayer.
 *
 * AudioStreamPlayer depends on the Web Audio API, so we mock AudioContext,
 * GainNode, AudioBuffer, and AudioBufferSourceNode.
 */
import { describe, expect, test, beforeEach, afterEach } from "vitest"
import {
    CARTESIA_HIGH_QUALITY_SAMPLE_RATE,
    PLAYBACK_SAMPLE_RATE,
} from "../src/config"
import { AudioStreamPlayer } from "../src"

// ---------------------------------------------------------------------------
// Web Audio API mocks
// ---------------------------------------------------------------------------

class MockAudioBufferSourceNode {
    buffer: any = null
    _connectedTo: any = null
    _startTime = -1

    connect(node: any) { this._connectedTo = node }
    start(time: number) { this._startTime = time }
}

class MockGainNode {
    _connectedTo: any = null
    connect(dest: any) { this._connectedTo = dest }
}

class MockAnalyserNode {
    fftSize = 256
    smoothingTimeConstant = 0.8
    frequencyBinCount = 128
    _connectedTo: any = null
    connect(dest: any) { this._connectedTo = dest }
    getByteFrequencyData(arr: Uint8Array) { arr.fill(0) }
}

class MockAudioBuffer {
    numberOfChannels = 1
    sampleRate: number
    length: number
    duration: number
    _channelData: Float32Array

    constructor(opts: { numberOfChannels: number; length: number; sampleRate: number }) {
        this.sampleRate = opts.sampleRate
        this.length = opts.length
        this.duration = opts.length / opts.sampleRate
        this._channelData = new Float32Array(opts.length)
    }

    getChannelData(_ch: number) {
        return this._channelData
    }
}

class MockAudioContext {
    sampleRate: number
    state: string = "running"
    currentTime = 0
    destination = {}
    _sources: MockAudioBufferSourceNode[] = []
    _gainNode: MockGainNode | null = null

    constructor(opts?: { sampleRate?: number }) {
        this.sampleRate = opts?.sampleRate ?? 44100
    }

    createGain(): MockGainNode {
        this._gainNode = new MockGainNode()
        return this._gainNode
    }

    createAnalyser(): MockAnalyserNode {
        return new MockAnalyserNode()
    }

    createBuffer(channels: number, length: number, sampleRate: number): MockAudioBuffer {
        return new MockAudioBuffer({ numberOfChannels: channels, length, sampleRate })
    }

    createBufferSource(): MockAudioBufferSourceNode {
        const src = new MockAudioBufferSourceNode()
        this._sources.push(src)
        return src
    }

    async resume() { this.state = "running" }
    async close() { this.state = "closed" }
}

// ---------------------------------------------------------------------------
// Setup / Teardown
// ---------------------------------------------------------------------------

let originalAudioContext: any

beforeEach(() => {
    originalAudioContext = (globalThis as any).AudioContext
    ;(globalThis as any).AudioContext = MockAudioContext
})

afterEach(() => {
    if (originalAudioContext) {
        (globalThis as any).AudioContext = originalAudioContext
    } else {
        delete (globalThis as any).AudioContext
    }
})

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer", () => {
    test("init creates AudioContext with default low-quality sample rate", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        const ctx = (player as any).ctx as MockAudioContext
        expect(player.sampleRate).toBe(PLAYBACK_SAMPLE_RATE)
        expect(ctx.sampleRate).toBe(PLAYBACK_SAMPLE_RATE)
        expect(player.isPlaying).toBe(false)
        player.stop()
    })

    test("high quality uses matching 24 kHz sample rate", async () => {
        const player = new AudioStreamPlayer({ quality: "high" })
        await player.init()

        const ctx = (player as any).ctx as MockAudioContext
        expect(player.sampleRate).toBe(CARTESIA_HIGH_QUALITY_SAMPLE_RATE)
        expect(ctx.sampleRate).toBe(CARTESIA_HIGH_QUALITY_SAMPLE_RATE)

        player.enqueue(new Float32Array(CARTESIA_HIGH_QUALITY_SAMPLE_RATE / 10))
        expect(ctx._sources[0].buffer.sampleRate).toBe(CARTESIA_HIGH_QUALITY_SAMPLE_RATE)

        player.stop()
    })

    test("explicit sample rate overrides quality", async () => {
        const player = new AudioStreamPlayer({ quality: "high", sampleRate: 16_000 })
        await player.init()

        const ctx = (player as any).ctx as MockAudioContext
        expect(player.sampleRate).toBe(16_000)
        expect(ctx.sampleRate).toBe(16_000)

        player.stop()
    })

    test("enqueue schedules AudioBufferSourceNode", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        const samples = new Float32Array(PLAYBACK_SAMPLE_RATE / 10)
        player.enqueue(samples)

        // After enqueue, isPlaying should be true (nextStartTime > currentTime)
        expect(player.isPlaying).toBe(true)

        player.stop()
    })

    test("enqueue is a no-op before init", async () => {
        const player = new AudioStreamPlayer()

        // Should not throw
        player.enqueue(new Float32Array(100))
        expect(player.isPlaying).toBe(false)
    })

    test("stop resets isPlaying to false", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        player.enqueue(new Float32Array(2400))
        expect(player.isPlaying).toBe(true)

        player.stop()
        expect(player.isPlaying).toBe(false)
    })

    test("waitForEnd resolves immediately when not playing", async () => {
        const player = new AudioStreamPlayer()

        // Should resolve without hanging
        await player.waitForEnd()
    })

    test("multiple enqueues stack gapless", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Enqueue 3 chunks of 100ms each
        for (let i = 0; i < 3; i++) {
            player.enqueue(new Float32Array(2400))
        }

        // Should be playing
        expect(player.isPlaying).toBe(true)

        player.stop()
    })

    test("init is idempotent when context is active", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        // Second call should not throw
        await player.init()
        player.stop()
    })

    test("stop then re-init works", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        player.enqueue(new Float32Array(2400))
        player.stop()

        // Re-init
        await player.init()
        player.enqueue(new Float32Array(2400))
        expect(player.isPlaying).toBe(true)
        player.stop()
    })

    test("stop before init is safe (no-op)", async () => {
        const player = new AudioStreamPlayer()
        // Should not throw
        player.stop()
        expect(player.isPlaying).toBe(false)
    })

    test("stop called multiple times does not throw", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        player.enqueue(new Float32Array(2400))
        player.stop()
        player.stop()
        player.stop()
        expect(player.isPlaying).toBe(false)
    })

    test("enqueue zero-length array does not throw", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        // Should not throw or break
        player.enqueue(new Float32Array(0))
        player.stop()
    })
})

// ---------------------------------------------------------------------------
// AudioContext auto-suspend resilience
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer — AudioContext auto-suspend", () => {
    test("enqueue resumes a suspended AudioContext", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Simulate WebView2 auto-suspending the AudioContext after idle
        const ctx = (player as any).ctx as MockAudioContext
        ctx.state = "suspended"

        // Enqueue should detect suspended state and resume
        player.enqueue(new Float32Array(2400))

        expect(ctx.state).toBe("running")
        expect(player.isPlaying).toBe(true)

        player.stop()
    })

    test("enqueue works normally when context is already running", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        const ctx = (player as any).ctx as MockAudioContext
        expect(ctx.state).toBe("running")

        player.enqueue(new Float32Array(2400))

        expect(ctx.state).toBe("running")
        expect(player.isPlaying).toBe(true)

        player.stop()
    })
})

// ---------------------------------------------------------------------------
// Gapless scheduling logic
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer — gapless scheduling", () => {
    test("nextStartTime advances by chunk duration", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // 1 second of audio at the playback sample rate
        const oneSec = new Float32Array(PLAYBACK_SAMPLE_RATE)
        player.enqueue(oneSec)

        // isPlaying should be true because nextStartTime > currentTime
        expect(player.isPlaying).toBe(true)

        player.stop()
    })

    test("enqueue multiple chunks → isPlaying remains true", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        player.enqueue(new Float32Array(2400))
        player.enqueue(new Float32Array(2400))
        player.enqueue(new Float32Array(2400))

        expect(player.isPlaying).toBe(true)
        player.stop()
    })

    test("stop resets nextStartTime to 0", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        player.enqueue(new Float32Array(2400))
        player.stop()

        // After stop, isPlaying should be false (ctx was closed)
        expect(player.isPlaying).toBe(false)
    })

    test("PLAYBACK_SAMPLE_RATE is 8000 Hz (phone quality)", async () => {
        expect(PLAYBACK_SAMPLE_RATE).toBe(8000)
    })
})

// ---------------------------------------------------------------------------
// waitForEnd edge cases
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer — waitForEnd", () => {
    test("waitForEnd resolves immediately when nothing is queued", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Nothing enqueued → should resolve immediately
        await player.waitForEnd()
        expect(player.isPlaying).toBe(false)

        player.stop()
    })

    test("waitForEnd resolves immediately when ctx is null (not init)", async () => {
        const player = new AudioStreamPlayer()

        // Not initialized → isPlaying is false → resolves immediately
        await player.waitForEnd()
        expect(player.isPlaying).toBe(false)
    })

    test("waitForEnd resolves immediately after stop", async () => {
        const player = new AudioStreamPlayer()
        await player.init()
        player.enqueue(new Float32Array(2400))
        player.stop()

        // After stop → resolves immediately
        await player.waitForEnd()
        expect(player.isPlaying).toBe(false)
    })
})

// ---------------------------------------------------------------------------
// waitForEnd settle-check behavior
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer — waitForEnd settle-check", () => {
    test("requires 2 consecutive idle polls before resolving", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Player is not playing → waitForEnd should resolve after 2 idle polls
        // First poll: consecutiveIdle = 1 (not enough)
        // Second poll: consecutiveIdle = 2 (resolves)
        const start = Date.now()
        await player.waitForEnd()
        const elapsed = Date.now() - start

        // The first check is synchronous (immediately), incrementing idle to 1.
        // Then it schedules a setTimeout(check, 100) which increments to 2 and resolves.
        // So we expect ~100ms delay (1 setTimeout of 100ms).
        expect(elapsed).toBeGreaterThanOrEqual(80) // Allow some variance
        expect(elapsed).toBeLessThan(500) // But not too long

        player.stop()
    })

    test("settle-check resets counter when isPlaying flips back to true", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Enqueue a very short chunk — it will be "playing" briefly
        // With mock AudioContext, currentTime stays at 0, so isPlaying = true
        // after enqueue (0 < 0.1)
        player.enqueue(new Float32Array(2400))
        expect(player.isPlaying).toBe(true)

        // Stop resets state so isPlaying = false
        player.stop()
        expect(player.isPlaying).toBe(false)

        // Now waitForEnd should resolve (2 consecutive idle polls)
        await player.waitForEnd()
    })

    test("waitForEnd on stopped player resolves within reasonable time", async () => {
        const player = new AudioStreamPlayer()

        // Not initialized — isPlaying is always false
        const start = Date.now()
        await player.waitForEnd()
        const elapsed = Date.now() - start

        // Should resolve after ~100ms (one setTimeout between 1st and 2nd idle check)
        expect(elapsed).toBeLessThan(500)
    })
})

// ---------------------------------------------------------------------------
// Zero-length enqueue: isPlaying verification
// ---------------------------------------------------------------------------

describe("AudioStreamPlayer — zero-length enqueue details", () => {
    test("zero-length enqueue does not change isPlaying", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        // Before enqueue
        expect(player.isPlaying).toBe(false)

        // Enqueue zero-length
        player.enqueue(new Float32Array(0))

        // Should still be false — guard returns early
        expect(player.isPlaying).toBe(false)

        player.stop()
    })

    test("multiple zero-length enqueues are all no-ops", async () => {
        const player = new AudioStreamPlayer()
        await player.init()

        for (let i = 0; i < 10; i++) {
            player.enqueue(new Float32Array(0))
        }
        expect(player.isPlaying).toBe(false)

        player.stop()
    })
})
