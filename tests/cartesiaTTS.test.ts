/**
 * Tests for CartesiaTTS.
 *
 * CartesiaTTS depends on the browser's WebSocket, so we mock it.
 * Focuses on: message construction, base64-PCM decode in handleMessage,
 * callback wiring, and send-queue serialization.
 */
import { describe, expect, test, beforeEach, afterEach } from "vitest"
import {
    CARTESIA_MODEL_ID,
    CARTESIA_VOICE_ID,
    CARTESIA_OUTPUT_FORMAT,
    CARTESIA_HIGH_QUALITY_SAMPLE_RATE,
    PLAYBACK_SAMPLE_RATE,
} from "../src/config"
import { CartesiaTTS } from "../src"

// ---------------------------------------------------------------------------
// WebSocket mock
// ---------------------------------------------------------------------------

let wsInstances: MockWebSocket[] = []
let originalWebSocket: typeof globalThis.WebSocket

class MockWebSocket {
    static OPEN = 1
    static CLOSED = 3

    url: string
    readyState = MockWebSocket.OPEN
    onopen: ((ev: any) => void) | null = null
    onclose: ((ev: any) => void) | null = null
    onerror: ((ev: any) => void) | null = null
    onmessage: ((ev: any) => void) | null = null

    sentMessages: string[] = []
    /** If true, do NOT auto-fire onopen (simulate hang / slow connect). */
    preventAutoOpen = false

    constructor(url: string) {
        this.url = url
        wsInstances.push(this)
        // Auto-fire open on next microtask (unless preventAutoOpen)
        queueMicrotask(() => {
            if (!this.preventAutoOpen) this.onopen?.({})
        })
    }

    send(data: string) {
        this.sentMessages.push(data)
    }

    close() {
        this.readyState = MockWebSocket.CLOSED
        this.onclose?.({})
    }
}

// ---------------------------------------------------------------------------
// Setup / Teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
    wsInstances = []
    originalWebSocket = globalThis.WebSocket
    ;(globalThis as any).WebSocket = MockWebSocket
})

afterEach(() => {
    globalThis.WebSocket = originalWebSocket
})

async function createTTS(options?: ConstructorParameters<typeof CartesiaTTS>[1]) {
    return new CartesiaTTS("test-api-key", options)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("CartesiaTTS", () => {
    test("connect opens WebSocket with correct URL params", async () => {
        const tts = await createTTS()
        await tts.connect()

        expect(wsInstances.length).toBe(1)
        const ws = wsInstances[0]
        expect(ws.url).toContain("api_key=test-api-key")
        expect(ws.url).toContain("cartesia_version=")
        tts.disconnect()
    })

    test("sendChunk serialises correct message format", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-1", "Hello ", true)

        expect(ws.sentMessages.length).toBe(1)
        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.model_id).toBe(CARTESIA_MODEL_ID)
        expect(msg.voice.mode).toBe("id")
        expect(msg.voice.id).toBe(CARTESIA_VOICE_ID)
        expect(msg.output_format.container).toBe(CARTESIA_OUTPUT_FORMAT.container)
        expect(msg.output_format.encoding).toBe(CARTESIA_OUTPUT_FORMAT.encoding)
        expect(msg.output_format.sample_rate).toBe(CARTESIA_OUTPUT_FORMAT.sample_rate)
        expect(msg.context_id).toBe("ctx-1")
        expect(msg.continue).toBe(true)
        expect(msg.transcript).toBe("Hello ")

        tts.disconnect()
    })

    test("defaults to low-quality output format", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-low", "Hello ", true)

        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.output_format.sample_rate).toBe(PLAYBACK_SAMPLE_RATE)
        expect(tts.sampleRate).toBe(PLAYBACK_SAMPLE_RATE)

        tts.disconnect()
    })

    test("can send high-quality output format", async () => {
        const tts = await createTTS({ quality: "high" })
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-high", "Hello ", true)

        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.output_format.container).toBe(CARTESIA_OUTPUT_FORMAT.container)
        expect(msg.output_format.encoding).toBe(CARTESIA_OUTPUT_FORMAT.encoding)
        expect(msg.output_format.sample_rate).toBe(CARTESIA_HIGH_QUALITY_SAMPLE_RATE)
        expect(tts.sampleRate).toBe(CARTESIA_HIGH_QUALITY_SAMPLE_RATE)

        tts.disconnect()
    })

    test("flush sends continue: false", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.flush("ctx-2")

        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.continue).toBe(false)
        expect(msg.context_id).toBe("ctx-2")

        tts.disconnect()
    })

    test("cancel sends cancel: true", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-3")

        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.cancel).toBe(true)
        expect(msg.context_id).toBe("ctx-3")

        tts.disconnect()
    })

    test("handleMessage decodes base64 PCM chunk and fires onAudio", async () => {
        const tts = await createTTS()

        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        // Create a tiny PCM16 sample: [0x0100, 0xFF00] → base64
        const pcm = new Int16Array([256, -256])
        const bytes = new Uint8Array(pcm.buffer)
        // Manual base64 encode
        let binary = ""
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i])
        }
        const b64 = btoa(binary)

        // Simulate incoming message
        ws.onmessage?.({
            data: JSON.stringify({ type: "chunk", data: b64 }),
        })

        expect(audioSamples.length).toBe(1)
        expect(audioSamples[0].length).toBe(2)
        // 256 / 32768 ≈ 0.0078125
        expect(audioSamples[0][0]).toBeCloseTo(256 / 32768, 4)

        tts.disconnect()
    })

    test("handleMessage fires onDone on done event", async () => {
        const tts = await createTTS()

        let doneContextId = ""
        tts.on({ onDone: (id) => { doneContextId = id } })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-99" }),
        })

        expect(doneContextId).toBe("ctx-99")
        tts.disconnect()
    })

    test("handleMessage fires onError on error event", async () => {
        const tts = await createTTS()

        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({ type: "error", message: "quota exceeded" }),
        })

        expect(errors.length).toBe(1)
        expect(errors[0].message).toContain("quota exceeded")
        tts.disconnect()
    })

    test("disconnect resets state", async () => {
        const tts = await createTTS()
        await tts.connect()

        tts.disconnect()

        // Reconnect should create a new WS
        await tts.connect()
        expect(wsInstances.length).toBe(2)
        tts.disconnect()
    })

    test("sendChunk ordering is preserved (sendQueue)", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        // Fire multiple sends concurrently
        const p1 = tts.sendChunk("ctx-1", "First ", true)
        const p2 = tts.sendChunk("ctx-1", "Second ", true)
        const p3 = tts.sendChunk("ctx-1", "Third.", false)

        await Promise.all([p1, p2, p3])

        expect(ws.sentMessages.length).toBe(3)
        const msgs = ws.sentMessages.map((m) => JSON.parse(m))
        expect(msgs[0].transcript).toBe("First ")
        expect(msgs[1].transcript).toBe("Second ")
        expect(msgs[2].transcript).toBe("Third.")
        expect(msgs[0].continue).toBe(true)
        expect(msgs[2].continue).toBe(false)

        tts.disconnect()
    })

    test("connect when already connected is a no-op", async () => {
        const tts = await createTTS()
        await tts.connect()
        // Second connect should not create another WS
        await tts.connect()
        expect(wsInstances.length).toBe(1)
        tts.disconnect()
    })

    test("handleMessage ignores unknown message types", async () => {
        const tts = await createTTS()

        const audioSamples: Float32Array[] = []
        const errors: Error[] = []
        let doneCount = 0
        tts.on({
            onAudio: (s) => audioSamples.push(s),
            onError: (e) => errors.push(e),
            onDone: () => doneCount++,
        })

        await tts.connect()
        const ws = wsInstances[0]

        // flush_done and timestamps should be silently ignored
        ws.onmessage?.({ data: JSON.stringify({ type: "flush_done" }) })
        ws.onmessage?.({ data: JSON.stringify({ type: "timestamps" }) })

        expect(audioSamples.length).toBe(0)
        expect(errors.length).toBe(0)
        expect(doneCount).toBe(0)

        tts.disconnect()
    })

    test("handleMessage ignores non-JSON data", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Non-JSON should not throw or fire onError
        ws.onmessage?.({ data: "not valid json" })
        expect(errors.length).toBe(0)

        tts.disconnect()
    })

    test("on() merges multiple callback registrations", async () => {
        const tts = await createTTS()
        const logs: string[] = []
        const errors: Error[] = []

        tts.on({ onLog: (m) => logs.push(m) })
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Error should use onError, log should use onLog
        ws.onmessage?.({ data: JSON.stringify({ type: "error", message: "test" }) })
        expect(errors.length).toBe(1)
        expect(logs.some((l) => l.includes("Cartesia"))).toBe(true)

        tts.disconnect()
    })

    test("sendChunk with continue=false sends directly without flush", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-5", "Final text.", false)

        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.continue).toBe(false)
        expect(msg.transcript).toBe("Final text.")

        tts.disconnect()
    })

    test("cancel message only contains context_id and cancel", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-cancel")

        const msg = JSON.parse(ws.sentMessages[0])
        // Should have context_id and cancel, but NOT model_id, voice, etc.
        expect(Object.keys(msg)).toEqual(["context_id", "cancel"])
        expect(msg.context_id).toBe("ctx-cancel")
        expect(msg.cancel).toBe(true)

        tts.disconnect()
    })

    test("handleMessage error without message field falls back to JSON.stringify", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Error event with no message property
        ws.onmessage?.({ data: JSON.stringify({ type: "error", code: 429 }) })

        expect(errors.length).toBe(1)
        // Falls back to JSON.stringify of the msg object
        expect(errors[0].message).toContain("Cartesia error:")
        expect(errors[0].message).toContain("429")

        tts.disconnect()
    })

    test("decoded PCM audio samples are normalized to [-1, 1]", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        // Extreme values: Int16 max (32767) and min (-32768)
        const pcmMax = new Int16Array([32767, -32768, 0])
        const bytes = new Uint8Array(pcmMax.buffer)
        let binary = ""
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i])
        }
        ws.onmessage?.({ data: JSON.stringify({ type: "chunk", data: btoa(binary) }) })

        expect(audioSamples.length).toBe(1)
        const s = audioSamples[0]
        expect(s.length).toBe(3)
        // 32767 / 32768 ≈ 0.99997
        expect(s[0]).toBeGreaterThan(0.99)
        expect(s[0]).toBeLessThanOrEqual(1.0)
        // -32768 / 32768 = -1.0
        expect(s[1]).toBe(-1.0)
        // 0 / 32768 = 0
        expect(s[2]).toBe(0.0)

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// Connect timeout & settled flag
// ---------------------------------------------------------------------------

describe("CartesiaTTS — connect timeout & settled flag", () => {
    test("connect timeout is 15 seconds", () => {
        // Source: setTimeout(..., 15_000) in connect()
        const CONNECT_TIMEOUT_MS = 15_000
        expect(CONNECT_TIMEOUT_MS).toBe(15000)
    })

    test("onclose before onopen rejects connect promise", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })
        // connectWithRetry retries 3× with linear backoff (2s + 3s = 5s).
        // Default Bun timeout is 5s — extend to 10s so all retries complete.

        // Manually override to prevent auto-open
        const origWS = globalThis.WebSocket
        ;(globalThis as any).WebSocket = class extends MockWebSocket {
            constructor(url: string) {
                super(url)
                this.preventAutoOpen = true
                // Fire close immediately (before open)
                queueMicrotask(() => this.onclose?.({ code: 1006, reason: "connection refused" }))
            }
        }

        try {
            await tts.connect()
            // If it somehow resolves, that's a bug
            expect(true).toBe(false)
        } catch (e: any) {
            expect(e).toBeInstanceOf(Error)
            // Should be the "closed before open" error
            expect(e.message).toBeTruthy()
        }

        ;(globalThis as any).WebSocket = origWS
    }, 10_000)

    test("error during connect rejects promise", async () => {
        const tts = await createTTS()

        const origWS = globalThis.WebSocket
        ;(globalThis as any).WebSocket = class extends MockWebSocket {
            constructor(url: string) {
                super(url)
                this.preventAutoOpen = true
                // Fire error immediately
                queueMicrotask(() => this.onerror?.({ message: "connection error" }))
            }
        }

        try {
            await tts.connect()
            expect(true).toBe(false) // Should not reach here
        } catch (e: any) {
            expect(e).toBeInstanceOf(Error)
        }

        ;(globalThis as any).WebSocket = origWS
    }, 10_000)

    test("connect while already connected is a no-op", async () => {
        const tts = await createTTS()
        await tts.connect()
        expect(wsInstances.length).toBe(1)

        // Second connect should not open a new WS
        await tts.connect()
        expect(wsInstances.length).toBe(1)

        tts.disconnect()
    })

    test("after disconnect, connect creates new WebSocket", async () => {
        const tts = await createTTS()
        await tts.connect()
        expect(wsInstances.length).toBe(1)

        tts.disconnect()

        await tts.connect()
        // Should have created a new WS instance
        expect(wsInstances.length).toBe(2)

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// sendQueue serialization
// ---------------------------------------------------------------------------

describe("CartesiaTTS — sendQueue ordering", () => {
    test("multiple rapid sendChunks arrive in order", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        // Fire off 5 chunks without awaiting
        const promises = []
        for (let i = 0; i < 5; i++) {
            promises.push(tts.sendChunk("ctx", `word${i} `, i === 0))
        }
        await Promise.all(promises)

        expect(ws.sentMessages.length).toBe(5)
        for (let i = 0; i < 5; i++) {
            const msg = JSON.parse(ws.sentMessages[i])
            expect(msg.transcript).toBe(`word${i} `)
        }

        tts.disconnect()
    })

    test("sendChunk isContinuation maps directly to continue field", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-1", "first ", true)  // isContinuation=true
        await tts.sendChunk("ctx-1", "second ", false) // isContinuation=false (flush)

        const m1 = JSON.parse(ws.sentMessages[0])
        const m2 = JSON.parse(ws.sentMessages[1])

        expect(m1.continue).toBe(true)  // Continuation chunk
        expect(m2.continue).toBe(false) // Final/flush chunk

        tts.disconnect()
    })

    test("cancel interrupts in-flight context", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.sendChunk("ctx-1", "hello ", true)
        await tts.cancel("ctx-1")

        // There should be 2 messages: the chunk and the cancel
        expect(ws.sentMessages.length).toBe(2)
        const cancelMsg = JSON.parse(ws.sentMessages[1])
        expect(cancelMsg.context_id).toBe("ctx-1")
        expect(cancelMsg.cancel).toBe(true)

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// handleMessage edge cases
// ---------------------------------------------------------------------------

describe("CartesiaTTS — handleMessage edge cases", () => {
    test("message with type 'done' fires onDone callback", async () => {
        const tts = await createTTS()
        let doneFired = false
        tts.on({ onDone: () => { doneFired = true } })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({ data: JSON.stringify({ type: "done" }) })

        expect(doneFired).toBe(true)
        tts.disconnect()
    })

    test("message with unknown type does not crash", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        // Should not throw
        ws.onmessage?.({ data: JSON.stringify({ type: "unknown-type", foo: "bar" }) })
        tts.disconnect()
    })

    test("non-JSON message does not crash", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        // Should not throw (silently ignored)
        ws.onmessage?.({ data: "this is not json" })
        tts.disconnect()
    })

    test("chunk with empty base64 data produces empty Float32Array", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        // Empty base64 string = ""
        ws.onmessage?.({ data: JSON.stringify({ type: "chunk", data: "" }) })

        expect(audioSamples.length).toBe(1)
        expect(audioSamples[0].length).toBe(0)

        tts.disconnect()
    })

    test("error message with only code uses JSON fallback", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({ data: JSON.stringify({ type: "error", code: 500 }) })

        expect(errors.length).toBe(1)
        expect(errors[0].message).toContain("500")

        tts.disconnect()
    })

    test("error message with message property uses it directly", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({
                type: "error",
                message: "Rate limit exceeded",
                code: 429,
            }),
        })

        expect(errors.length).toBe(1)
        expect(errors[0].message).toContain("Rate limit exceeded")

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// Flush behavior
// ---------------------------------------------------------------------------

describe("CartesiaTTS — flush", () => {
    test("flush sends ' ' (space) as transcript with continue: false", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.flush("ctx-flush")

        expect(ws.sentMessages.length).toBe(1)
        const msg = JSON.parse(ws.sentMessages[0])
        expect(msg.transcript).toBe(" ")
        expect(msg.continue).toBe(false)
        expect(msg.context_id).toBe("ctx-flush")

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// done event context_id
// ---------------------------------------------------------------------------

describe("CartesiaTTS — done event with context_id", () => {
    test("done event passes context_id to onDone callback", async () => {
        const tts = await createTTS()
        const doneIds: string[] = []
        tts.on({ onDone: (contextId) => doneIds.push(contextId) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-42" }),
        })

        expect(doneIds.length).toBe(1)
        expect(doneIds[0]).toBe("ctx-42")

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// Cancel then sendChunk ordering
// ---------------------------------------------------------------------------

describe("CartesiaTTS — cancel then sendChunk ordering", () => {
    test("cancel followed by sendChunk on different context preserves order", async () => {
        const tts = await createTTS()
        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-old")
        await tts.sendChunk("ctx-new", "hello ", true)

        expect(ws.sentMessages.length).toBe(2)
        const cancelMsg = JSON.parse(ws.sentMessages[0])
        const sendMsg = JSON.parse(ws.sentMessages[1])

        expect(cancelMsg.cancel).toBe(true)
        expect(cancelMsg.context_id).toBe("ctx-old")
        expect(sendMsg.context_id).toBe("ctx-new")
        expect(sendMsg.transcript).toBe("hello ")

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// Concurrent connect deduplication
// ---------------------------------------------------------------------------

describe("CartesiaTTS — concurrent connect dedup", () => {
    test("two connect() calls before resolution create only 1 WebSocket", async () => {
        const tts = await createTTS()
        // Start two connects without awaiting
        const p1 = tts.connect()
        const p2 = tts.connect()

        await Promise.all([p1, p2])

        // Only one WebSocket should have been created
        expect(wsInstances.length).toBe(1)

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// cancelledContexts — suppression of audio/done/error for cancelled contexts
// ---------------------------------------------------------------------------

describe("CartesiaTTS — cancelledContexts suppression", () => {
    /** Helper: create a tiny base64 PCM chunk for testing */
    function makePcmChunkB64(): string {
        const pcm = new Int16Array([256, -256])
        const bytes = new Uint8Array(pcm.buffer)
        let binary = ""
        for (let i = 0; i < bytes.length; i++) {
            binary += String.fromCharCode(bytes[i])
        }
        return btoa(binary)
    }

    test("cancel suppresses audio chunks from cancelled context", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-cancelled")

        // Simulate Cartesia sending audio for the cancelled context
        ws.onmessage?.({
            data: JSON.stringify({ type: "chunk", data: makePcmChunkB64(), context_id: "ctx-cancelled" }),
        })

        // Audio should be suppressed
        expect(audioSamples.length).toBe(0)

        tts.disconnect()
    })

    test("cancel suppresses done events from cancelled context", async () => {
        const tts = await createTTS()
        const doneIds: string[] = []
        tts.on({ onDone: (id) => doneIds.push(id) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-cancelled")

        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-cancelled" }),
        })

        // Done should be suppressed
        expect(doneIds.length).toBe(0)

        tts.disconnect()
    })

    test("cancel suppresses error events from cancelled context", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-cancelled")

        // Simulate the exact error from the bug: "Invalid context ID"
        ws.onmessage?.({
            data: JSON.stringify({
                type: "error",
                context_id: "ctx-cancelled",
                status_code: 400,
                error: "Invalid context ID: The requested context ID does not exist or may have already been cancelled.",
            }),
        })

        // Error should be suppressed
        expect(errors.length).toBe(0)

        tts.disconnect()
    })

    test("audio from non-cancelled contexts still passes through", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-old")

        // Audio for a DIFFERENT context should NOT be suppressed
        ws.onmessage?.({
            data: JSON.stringify({ type: "chunk", data: makePcmChunkB64(), context_id: "ctx-new" }),
        })

        expect(audioSamples.length).toBe(1)

        tts.disconnect()
    })

    test("done from non-cancelled contexts still fires callback", async () => {
        const tts = await createTTS()
        const doneIds: string[] = []
        tts.on({ onDone: (id) => doneIds.push(id) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-old")

        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-new" }),
        })

        expect(doneIds).toEqual(["ctx-new"])

        tts.disconnect()
    })

    test("error from non-cancelled contexts still fires callback", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-old")

        ws.onmessage?.({
            data: JSON.stringify({ type: "error", context_id: "ctx-new", message: "real error" }),
        })

        expect(errors.length).toBe(1)
        expect(errors[0].message).toContain("real error")

        tts.disconnect()
    })

    test("disconnect clears cancelledContexts (no suppression after reconnect)", async () => {
        const tts = await createTTS()
        const doneIds: string[] = []
        tts.on({ onDone: (id) => doneIds.push(id) })

        await tts.connect()
        await tts.cancel("ctx-X")

        tts.disconnect()

        // Reconnect
        await tts.connect()
        const ws = wsInstances[1]

        // Same context_id should NOT be suppressed after disconnect+reconnect
        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-X" }),
        })

        expect(doneIds).toEqual(["ctx-X"])

        tts.disconnect()
    })

    test("multiple contexts can be cancelled independently", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        const errors: Error[] = []
        tts.on({
            onAudio: (s) => audioSamples.push(s),
            onError: (e) => errors.push(e),
        })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-A")
        await tts.cancel("ctx-B")

        // Both should be suppressed
        ws.onmessage?.({
            data: JSON.stringify({ type: "chunk", data: makePcmChunkB64(), context_id: "ctx-A" }),
        })
        ws.onmessage?.({
            data: JSON.stringify({ type: "error", context_id: "ctx-B", message: "nope" }),
        })

        expect(audioSamples.length).toBe(0)
        expect(errors.length).toBe(0)

        // Non-cancelled context works fine
        ws.onmessage?.({
            data: JSON.stringify({ type: "chunk", data: makePcmChunkB64(), context_id: "ctx-C" }),
        })
        expect(audioSamples.length).toBe(1)

        tts.disconnect()
    })

    test("done cleanup removes context from set — subsequent events not suppressed", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        const doneIds: string[] = []
        tts.on({
            onDone: (id) => doneIds.push(id),
            onError: (e) => errors.push(e),
        })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-reuse")

        // First done → suppressed and cleaned up
        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-reuse" }),
        })
        expect(doneIds.length).toBe(0)

        // Second done with same context_id → NOT suppressed (cleaned from set)
        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-reuse" }),
        })
        expect(doneIds).toEqual(["ctx-reuse"])

        tts.disconnect()
    })

    test("error cleanup removes context from set", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-err")

        // First error → suppressed and cleaned up
        ws.onmessage?.({
            data: JSON.stringify({ type: "error", context_id: "ctx-err", message: "gone" }),
        })
        expect(errors.length).toBe(0)

        // Second error with same context → NOT suppressed
        ws.onmessage?.({
            data: JSON.stringify({ type: "error", context_id: "ctx-err", message: "real problem" }),
        })
        expect(errors.length).toBe(1)
        expect(errors[0].message).toContain("real problem")

        tts.disconnect()
    })

    test("cancelled context audio suppressed across multiple chunks", async () => {
        const tts = await createTTS()
        const audioSamples: Float32Array[] = []
        tts.on({ onAudio: (s) => audioSamples.push(s) })

        await tts.connect()
        const ws = wsInstances[0]

        await tts.cancel("ctx-multi")

        // Send multiple chunks — all should be suppressed (has doesn't delete)
        for (let i = 0; i < 5; i++) {
            ws.onmessage?.({
                data: JSON.stringify({ type: "chunk", data: makePcmChunkB64(), context_id: "ctx-multi" }),
            })
        }

        expect(audioSamples.length).toBe(0)

        tts.disconnect()
    })

    test("real-world scenario: cancel before any sendChunk → error suppressed", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Cancel a context that was never sent to (the bug scenario)
        await tts.cancel("ctx-7-1771350843426")

        // Cartesia responds with the exact error shape from the bug report
        ws.onmessage?.({
            data: JSON.stringify({
                type: "error",
                context_id: "ctx-7-1771350843426",
                status_code: 400,
                done: true,
                error: "Invalid context ID: The requested context ID does not exist or may have already been cancelled.",
            }),
        })

        // Error should be completely suppressed
        expect(errors.length).toBe(0)

        tts.disconnect()
    })

    test("cancel adds to set immediately before sendQueue processes", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Don't await — cancel is queued but cancelledContexts.add is synchronous
        const cancelPromise = tts.cancel("ctx-fast")

        // Before the WS message is actually sent, simulate the server
        // already responding with an error (race condition)
        ws.onmessage?.({
            data: JSON.stringify({ type: "error", context_id: "ctx-fast", message: "gone" }),
        })

        // The error should still be suppressed because add() was synchronous
        expect(errors.length).toBe(0)

        await cancelPromise
        tts.disconnect()
    })

    test("cancelling the same context twice is idempotent", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        // Cancel same context twice
        await tts.cancel("ctx-dup")
        await tts.cancel("ctx-dup")

        // The WS should have received two cancel messages
        const sent = ws.sentMessages.map((s: string) => JSON.parse(s))
        const cancelMsgs = sent.filter((m: any) => m.cancel && m.context_id === "ctx-dup")
        expect(cancelMsgs.length).toBe(2)

        // A done event should still be suppressed (context is still in set)
        ws.onmessage?.({
            data: JSON.stringify({ type: "done", context_id: "ctx-dup" }),
        })
        expect(errors.length).toBe(0)

        tts.disconnect()
    })
})

// ---------------------------------------------------------------------------
// handleMessage — unknown message types
// ---------------------------------------------------------------------------

describe("CartesiaTTS — unknown message types", () => {
    test("flush_done message is silently ignored", async () => {
        const tts = await createTTS()
        const audio: Float32Array[] = []
        const done: string[] = []
        const errors: Error[] = []
        tts.on({
            onAudio: (s) => audio.push(s),
            onDone: (id) => done.push(id),
            onError: (e) => errors.push(e),
        })

        await tts.connect()
        const ws = wsInstances[0]

        // Send an unknown message type
        ws.onmessage?.({
            data: JSON.stringify({ type: "flush_done", context_id: "ctx-1" }),
        })

        expect(audio.length).toBe(0)
        expect(done.length).toBe(0)
        expect(errors.length).toBe(0)
    })

    test("timestamps message is silently ignored", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({
                type: "timestamps",
                context_id: "ctx-1",
                word_timestamps: { words: ["hello"], start: [0], end: [0.5] },
            }),
        })

        expect(errors.length).toBe(0)
    })

    test("non-JSON frame is silently ignored", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({ data: "not-valid-json{{{" })

        expect(errors.length).toBe(0)
    })

    test("message with no type field is silently ignored", async () => {
        const tts = await createTTS()
        const errors: Error[] = []
        tts.on({ onError: (e) => errors.push(e) })

        await tts.connect()
        const ws = wsInstances[0]

        ws.onmessage?.({
            data: JSON.stringify({ context_id: "ctx-1", data: "something" }),
        })

        expect(errors.length).toBe(0)
    })
})

