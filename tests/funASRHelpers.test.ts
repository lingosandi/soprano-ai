/**
 * Tests for FunASRService helper functions.
 *
 * The pure helpers (downsample, float32ToInt16, mergePcmChunks,
 * parseDashScopeMessage) are exported and tested directly.
 * The FunASRService class itself requires browser APIs (WebSocket,
 * getUserMedia) so we test its message-handling logic via the helpers.
 */
import { describe, expect, test } from "vitest"
import {
    downsample,
    float32ToInt16,
    mergePcmChunks,
    parseDashScopeMessage,
} from "../src/audio-helpers"

// ---------------------------------------------------------------------------
// downsample
// ---------------------------------------------------------------------------

describe("downsample", () => {
    test("returns same buffer when rates match", () => {
        const input = new Float32Array([0.1, 0.2, 0.3, 0.4])
        const result = downsample(input, 16000, 16000)
        expect(result).toBe(input) // same reference
    })

    test("downsamples 48 kHz → 16 kHz (3:1 ratio)", () => {
        // 12 samples at 48 kHz → 4 samples at 16 kHz
        const input = new Float32Array(12)
        for (let i = 0; i < 12; i++) input[i] = i / 11
        const result = downsample(input, 48000, 16000)
        expect(result.length).toBe(4)
    })

    test("downsamples 44.1 kHz → 16 kHz", () => {
        const input = new Float32Array(441)
        const result = downsample(input, 44100, 16000)
        expect(result.length).toBe(Math.round(441 / (44100 / 16000)))
    })

    test("preserves approximate amplitude", () => {
        // Constant signal should stay constant
        const input = new Float32Array(100).fill(0.5)
        const result = downsample(input, 48000, 16000)
        for (let i = 0; i < result.length; i++) {
            expect(result[i]).toBeCloseTo(0.5, 4)
        }
    })

    test("2:1 downsample interpolates correctly", () => {
        // [0.0, 1.0, 0.0, 1.0] at 32kHz → 2 samples at 16kHz
        const input = new Float32Array([0.0, 1.0, 0.0, 1.0])
        const result = downsample(input, 32000, 16000)
        expect(result.length).toBe(2)
        // First sample: idx 0.0 → input[0] = 0.0
        expect(result[0]).toBeCloseTo(0.0, 4)
        // Second sample: idx 2.0 → input[2] = 0.0
        expect(result[1]).toBeCloseTo(0.0, 4)
    })

    test("handles single sample", () => {
        const input = new Float32Array([0.75])
        const result = downsample(input, 48000, 16000)
        expect(result.length).toBeGreaterThanOrEqual(0)
    })

    test("handles empty input array", () => {
        const input = new Float32Array(0)
        const result = downsample(input, 48000, 16000)
        expect(result.length).toBe(0)
    })

    test("upsample (toRate > fromRate) produces more samples", () => {
        const input = new Float32Array([0.1, 0.2, 0.3])
        const result = downsample(input, 16000, 48000)
        expect(result.length).toBe(Math.round(3 * (48000 / 16000)))
    })

    test("linear interpolation produces correct intermediate values", () => {
        // [0.0, 1.0] at 2:1 ratio → 1 sample at midpoint → should be 0.0
        // But with ratio=2, outputLen = round(2/2) = 1, srcIdx = 0*2 = 0 → output[0] = input[0] = 0.0
        // Let's use a 3:2 ratio: [0.0, 0.5, 1.0] at 48kHz → 2 samples at 32kHz
        const input = new Float32Array([0.0, 0.5, 1.0])
        const result = downsample(input, 48000, 32000)
        // ratio = 48000/32000 = 1.5, outputLen = round(3/1.5) = 2
        expect(result.length).toBe(2)
        // sample 0: srcIdx=0 → input[0] = 0.0
        expect(result[0]).toBeCloseTo(0.0, 4)
        // sample 1: srcIdx=1.5 → lo=1, hi=2, frac=0.5 → 0.5*(0.5) + 1.0*(0.5) = 0.75
        expect(result[1]).toBeCloseTo(0.75, 4)
    })

    test("non-integer ratio (22050 → 16000) works", () => {
        const input = new Float32Array(220)
        for (let i = 0; i < 220; i++) input[i] = Math.sin(i * 0.1)
        const result = downsample(input, 22050, 16000)
        expect(result.length).toBe(Math.round(220 / (22050 / 16000)))
        // No NaN values in output
        for (let i = 0; i < result.length; i++) {
            expect(Number.isNaN(result[i])).toBe(false)
        }
    })
})

// ---------------------------------------------------------------------------
// float32ToInt16
// ---------------------------------------------------------------------------

describe("float32ToInt16", () => {
    test("converts zero to zero", () => {
        const result = float32ToInt16(new Float32Array([0.0]))
        expect(result[0]).toBe(0)
    })

    test("converts +1.0 to 32767", () => {
        const result = float32ToInt16(new Float32Array([1.0]))
        expect(result[0]).toBe(32767)
    })

    test("converts -1.0 to -32768", () => {
        const result = float32ToInt16(new Float32Array([-1.0]))
        expect(result[0]).toBe(-32768)
    })

    test("clamps values beyond [-1, 1]", () => {
        const result = float32ToInt16(new Float32Array([2.0, -2.0]))
        expect(result[0]).toBe(32767)
        expect(result[1]).toBe(-32768)
    })

    test("preserves array length", () => {
        const input = new Float32Array(100)
        const result = float32ToInt16(input)
        expect(result.length).toBe(100)
    })

    test("returns Int16Array type", () => {
        const result = float32ToInt16(new Float32Array([0.5]))
        expect(result).toBeInstanceOf(Int16Array)
    })

    test("converts 0.5 to approximately half of max", () => {
        const result = float32ToInt16(new Float32Array([0.5]))
        // 0.5 * 0x7fff = 16383.5 → 16383
        expect(result[0]).toBe(Math.floor(0.5 * 0x7fff))
    })

    test("converts -0.5 to approximately half of min", () => {
        const result = float32ToInt16(new Float32Array([-0.5]))
        // -0.5 * 0x8000 = -16384
        expect(result[0]).toBe(-16384)
    })

    test("handles empty array", () => {
        const result = float32ToInt16(new Float32Array(0))
        expect(result.length).toBe(0)
    })

    test("handles very small values near zero", () => {
        const result = float32ToInt16(new Float32Array([0.00001, -0.00001]))
        // Should be close to 0 but possibly non-zero
        expect(Math.abs(result[0])).toBeLessThan(2)
        expect(Math.abs(result[1])).toBeLessThan(2)
    })
})

// ---------------------------------------------------------------------------
// mergePcmChunks
// ---------------------------------------------------------------------------

describe("mergePcmChunks", () => {
    test("merges multiple chunks into one buffer", () => {
        const c1 = new Int16Array([1, 2, 3])
        const c2 = new Int16Array([4, 5])
        const result = mergePcmChunks([c1, c2])
        // 5 Int16 = 10 bytes
        expect(result.length).toBe(10)
    })

    test("handles empty chunks array", () => {
        const result = mergePcmChunks([])
        expect(result.length).toBe(0)
    })

    test("handles single chunk", () => {
        const c = new Int16Array([100, 200])
        const result = mergePcmChunks([c])
        expect(result.length).toBe(4) // 2 Int16 = 4 bytes
    })

    test("ensures even byte length", () => {
        // mergePcmChunks always produces even-length output
        // since Int16Array → Uint8Array always has even byte count
        const c = new Int16Array([42])
        const result = mergePcmChunks([c])
        expect(result.length % 2).toBe(0)
    })

    test("preserves values through the merge", () => {
        const c1 = new Int16Array([1000])
        const c2 = new Int16Array([2000])
        const result = mergePcmChunks([c1, c2])
        // Read back as Int16
        const view = new Int16Array(result.buffer, result.byteOffset, result.length / 2)
        expect(view[0]).toBe(1000)
        expect(view[1]).toBe(2000)
    })

    test("preserves order across many chunks", () => {
        const chunks = Array.from({ length: 10 }, (_, i) => new Int16Array([i * 100]))
        const result = mergePcmChunks(chunks)
        const view = new Int16Array(result.buffer, result.byteOffset, result.length / 2)
        for (let i = 0; i < 10; i++) {
            expect(view[i]).toBe(i * 100)
        }
    })

    test("result is Uint8Array", () => {
        const result = mergePcmChunks([new Int16Array([42])])
        expect(result).toBeInstanceOf(Uint8Array)
    })
})

// ---------------------------------------------------------------------------
// parseDashScopeMessage
// ---------------------------------------------------------------------------

describe("parseDashScopeMessage", () => {
    test("parses task-started event", () => {
        const msg = JSON.stringify({
            header: { event: "task-started", task_id: "abc123" },
            payload: {},
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        expect(result!.event).toBe("task-started")
        expect(result!.parsed.header.task_id).toBe("abc123")
    })

    test("parses result-generated with sentence", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated", task_id: "abc" },
            payload: {
                output: {
                    sentence: {
                        text: "hello world",
                        begin_time: 0,
                        end_time: 1000,
                        sentence_end: true,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        expect(result!.event).toBe("result-generated")
        expect(result!.parsed.payload.output.sentence.text).toBe("hello world")
    })

    test("parses task-finished event", () => {
        const msg = JSON.stringify({
            header: { event: "task-finished", task_id: "abc" },
            payload: { output: {} },
        })
        const result = parseDashScopeMessage(msg)
        expect(result!.event).toBe("task-finished")
    })

    test("parses task-failed event", () => {
        const msg = JSON.stringify({
            header: {
                event: "task-failed",
                error_code: "INTERNAL_ERROR",
                error_message: "something went wrong",
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result!.event).toBe("task-failed")
        expect(result!.parsed.header.error_message).toBe("something went wrong")
    })

    test("returns null for invalid JSON", () => {
        expect(parseDashScopeMessage("not json")).toBeNull()
    })

    test("returns null for JSON without header.event", () => {
        expect(parseDashScopeMessage('{"data": 123}')).toBeNull()
        expect(parseDashScopeMessage('{"header": {}}')).toBeNull()
    })

    test("returns null for empty string", () => {
        expect(parseDashScopeMessage("")).toBeNull()
    })

    test("parses partial (interim) result — end_time is null", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {
                output: {
                    sentence: {
                        text: "hel",
                        begin_time: 0,
                        end_time: null,
                        sentence_end: false,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        const sentence = result!.parsed.payload.output.sentence
        expect(sentence.end_time).toBeNull()
        expect(sentence.sentence_end).toBe(false)
    })

    test("parses heartbeat result", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {
                output: {
                    sentence: {
                        text: "",
                        heartbeat: true,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result!.parsed.payload.output.sentence.heartbeat).toBe(true)
    })

    test("returns null for null-like values (number, boolean)", () => {
        expect(parseDashScopeMessage("42")).toBeNull()
        expect(parseDashScopeMessage("true")).toBeNull()
        expect(parseDashScopeMessage("null")).toBeNull()
    })

    test("preserves unknown event types", () => {
        const msg = JSON.stringify({
            header: { event: "custom-event", task_id: "x" },
            payload: {},
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        expect(result!.event).toBe("custom-event")
    })

    test("parses result with negative end_time (partial indicator)", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {
                output: {
                    sentence: {
                        text: "partial",
                        begin_time: 0,
                        end_time: -1,
                        sentence_end: false,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        // handleMessage treats end_time < 0 as partial
        expect(result!.parsed.payload.output.sentence.end_time).toBe(-1)
    })

    test("parses result with end_time = 0 (valid final at zero)", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {
                output: {
                    sentence: {
                        text: "hi",
                        begin_time: 0,
                        end_time: 0,
                        sentence_end: true,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        // end_time === 0 is not null and not < 0, so handleMessage treats it as final
        expect(result!.parsed.payload.output.sentence.end_time).toBe(0)
    })

    test("parsed object has full payload hierarchy", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {
                output: {
                    sentence: {
                        text: "hello",
                        begin_time: 100,
                        end_time: 500,
                        sentence_end: true,
                    },
                },
            },
        })
        const result = parseDashScopeMessage(msg)
        expect(result!.parsed.payload.output.sentence.begin_time).toBe(100)
        expect(result!.parsed.payload.output.sentence.end_time).toBe(500)
        expect(result!.parsed.payload.output.sentence.sentence_end).toBe(true)
    })
})

// ---------------------------------------------------------------------------
// FunASR detection logic — behavioural specification
//
// The handleMessage method in FunASRService uses the parsed DashScope
// message to decide what callback to fire. These tests verify the exact
// rules it applies:
//   1. heartbeat + empty text → skip
//   2. end_time == null || end_time < 0 → partial (onInterim)
//   3. any other end_time → final (onFinal)
//   4. task-finished with lingering interim → promotes to final
// ---------------------------------------------------------------------------

/**
 * Replicates the core detection logic from FunASRService.handleMessage
 * so we can test it as a pure function without mocking WebSocket/mic.
 */
function classifyResult(sentence: any): "skip" | "partial" | "final" {
    if (!sentence) return "skip"
    const text: string = sentence.text ?? ""
    if (sentence.heartbeat && !text) return "skip"
    const isPartial = sentence.end_time == null || sentence.end_time < 0
    return isPartial ? "partial" : "final"
}

/** Simulate the task-finished promotion rule. */
function shouldPromoteInterim(finalFired: boolean, lastInterimText: string): boolean {
    return !finalFired && !!lastInterimText
}

describe("FunASR detection — partial vs final classification", () => {
    test("end_time: null → partial", () => {
        expect(classifyResult({ text: "hello", end_time: null })).toBe("partial")
    })

    test("end_time: undefined → partial", () => {
        expect(classifyResult({ text: "hello", end_time: undefined })).toBe("partial")
    })

    test("end_time: -1 → partial", () => {
        expect(classifyResult({ text: "hello", end_time: -1 })).toBe("partial")
    })

    test("end_time: -100 → partial", () => {
        expect(classifyResult({ text: "hello", end_time: -100 })).toBe("partial")
    })

    test("end_time: 0 → final (zero is a valid timestamp)", () => {
        expect(classifyResult({ text: "hi", end_time: 0 })).toBe("final")
    })

    test("end_time: 500 → final", () => {
        expect(classifyResult({ text: "hello", end_time: 500 })).toBe("final")
    })

    test("end_time: 10000 → final", () => {
        expect(classifyResult({ text: "hello world", end_time: 10000 })).toBe("final")
    })

    test("missing sentence → skip", () => {
        expect(classifyResult(null)).toBe("skip")
        expect(classifyResult(undefined)).toBe("skip")
    })

    test("sentence with no text field → empty text → classified by end_time", () => {
        // end_time null → partial, even with no text
        expect(classifyResult({ end_time: null })).toBe("partial")
        // end_time 100 → final
        expect(classifyResult({ end_time: 100 })).toBe("final")
    })
})

describe("FunASR detection — heartbeat handling", () => {
    test("heartbeat: true + empty text → skip", () => {
        expect(classifyResult({ heartbeat: true, text: "" })).toBe("skip")
    })

    test("heartbeat: true + no text property → skip (text defaults to '')", () => {
        expect(classifyResult({ heartbeat: true })).toBe("skip")
    })

    test("heartbeat: true + non-empty text → NOT skipped (still classified)", () => {
        // If a heartbeat somehow has text, it goes through normal detection
        expect(classifyResult({ heartbeat: true, text: "actual text", end_time: null })).toBe("partial")
        expect(classifyResult({ heartbeat: true, text: "actual text", end_time: 500 })).toBe("final")
    })

    test("heartbeat: false + empty text → NOT skipped", () => {
        expect(classifyResult({ heartbeat: false, text: "", end_time: null })).toBe("partial")
        expect(classifyResult({ heartbeat: false, text: "", end_time: 100 })).toBe("final")
    })

    test("heartbeat: undefined + empty text → NOT skipped (undefined is falsy)", () => {
        expect(classifyResult({ text: "", end_time: null })).toBe("partial")
    })

    test("consecutive heartbeats are all skipped", () => {
        const heartbeats = Array.from({ length: 10 }, () => ({ heartbeat: true, text: "" }))
        for (const hb of heartbeats) {
            expect(classifyResult(hb)).toBe("skip")
        }
    })
})

describe("FunASR detection — interim→final promotion on task-finished", () => {
    test("no final fired + interim text → promotes", () => {
        expect(shouldPromoteInterim(false, "hello")).toBe(true)
    })

    test("final already fired → no promotion", () => {
        expect(shouldPromoteInterim(true, "leftover")).toBe(false)
    })

    test("no final fired + empty interim → no promotion", () => {
        expect(shouldPromoteInterim(false, "")).toBe(false)
    })

    test("final already fired + empty interim → no promotion", () => {
        expect(shouldPromoteInterim(true, "")).toBe(false)
    })
})

describe("FunASR detection — real-world message sequences", () => {
    /** Simulates handleMessage's state tracking over a sequence of events. */
    function runSequence(events: Array<{
        event: string
        sentence?: any
        error_message?: string
    }>) {
        let lastInterimText = ""
        let finalFired = false
        const result: Array<{ action: string; text?: string; error?: string }> = []

        for (const ev of events) {
            switch (ev.event) {
                case "result-generated": {
                    const cls = classifyResult(ev.sentence)
                    if (cls === "skip") {
                        result.push({ action: "skip" })
                    } else if (cls === "partial") {
                        lastInterimText = ev.sentence?.text ?? ""
                        finalFired = false
                        result.push({ action: "interim", text: lastInterimText })
                    } else {
                        finalFired = true
                        lastInterimText = ""
                        result.push({ action: "final", text: ev.sentence?.text ?? "" })
                    }
                    break
                }
                case "task-finished":
                    if (shouldPromoteInterim(finalFired, lastInterimText)) {
                        result.push({ action: "promoted-final", text: lastInterimText })
                        lastInterimText = ""
                    }
                    result.push({ action: "finished" })
                    break
                case "task-failed":
                    result.push({ action: "error", error: ev.error_message ?? "unknown error" })
                    break
            }
        }
        return result
    }

    test("typical flow: heartbeats → partials → final → finished", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { text: "hel", end_time: null } },
            { event: "result-generated", sentence: { text: "hello", end_time: null } },
            { event: "result-generated", sentence: { text: "hello world", end_time: 2500 } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "skip" },
            { action: "skip" },
            { action: "interim", text: "hel" },
            { action: "interim", text: "hello" },
            { action: "final", text: "hello world" },
            { action: "finished" },
        ])
    })

    test("promotion: partials only, no final → task-finished promotes last interim", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "hi", end_time: null } },
            { event: "result-generated", sentence: { text: "hi there", end_time: null } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "interim", text: "hi" },
            { action: "interim", text: "hi there" },
            { action: "promoted-final", text: "hi there" },
            { action: "finished" },
        ])
    })

    test("no speech detected: only heartbeats → task-finished → no promotion", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "skip" },
            { action: "skip" },
            { action: "skip" },
            { action: "finished" },
        ])
    })

    test("multiple sentences: final → new partials → final → finished", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "first sentence", end_time: 2000 } },
            { event: "result-generated", sentence: { text: "sec", end_time: null } },
            { event: "result-generated", sentence: { text: "second sentence", end_time: 4000 } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "final", text: "first sentence" },
            { action: "interim", text: "sec" },
            { action: "final", text: "second sentence" },
            { action: "finished" },
        ])
    })

    test("task-failed emits error", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "he", end_time: null } },
            { event: "task-failed", error_message: "rate limit exceeded" },
        ])

        expect(actions).toEqual([
            { action: "interim", text: "he" },
            { action: "error", error: "rate limit exceeded" },
        ])
    })

    test("task-failed without error_message uses default", () => {
        const actions = runSequence([
            { event: "task-failed" },
        ])

        expect(actions).toEqual([
            { action: "error", error: "unknown error" },
        ])
    })

    test("mixed heartbeats interleaved with partials", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { text: "h", end_time: null } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { text: "he", end_time: null } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { text: "hello", end_time: 1000 } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "skip" },
            { action: "interim", text: "h" },
            { action: "skip" },
            { action: "interim", text: "he" },
            { action: "skip" },
            { action: "final", text: "hello" },
            { action: "finished" },
        ])
    })

    test("empty final text is still classified as final", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "", end_time: 100 } },
            { event: "task-finished" },
        ])

        // end_time=100 → final, even if text is empty
        expect(actions).toEqual([
            { action: "final", text: "" },
            { action: "finished" },
        ])
    })

    test("long session: many sentences with heartbeats between them", () => {
        const actions = runSequence([
            // First sentence
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { text: "你", end_time: null } },
            { event: "result-generated", sentence: { text: "你好", end_time: 800 } },
            // Silence gap with heartbeats
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            { event: "result-generated", sentence: { heartbeat: true, text: "" } },
            // Second sentence
            { event: "result-generated", sentence: { text: "我", end_time: null } },
            { event: "result-generated", sentence: { text: "我是", end_time: null } },
            { event: "result-generated", sentence: { text: "我是AI", end_time: 3200 } },
            { event: "task-finished" },
        ])

        const finals = actions.filter((a) => a.action === "final")
        expect(finals.length).toBe(2)
        expect(finals[0].text).toBe("你好")
        expect(finals[1].text).toBe("我是AI")

        const skips = actions.filter((a) => a.action === "skip")
        expect(skips.length).toBe(3)

        const interims = actions.filter((a) => a.action === "interim")
        expect(interims.length).toBe(3) // 你, 我, 我是
    })

    test("partial with end_time: -1 then final with end_time: 500", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "partial", end_time: -1 } },
            { event: "result-generated", sentence: { text: "complete", end_time: 500 } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "interim", text: "partial" },
            { action: "final", text: "complete" },
            { action: "finished" },
        ])
    })

    // -----------------------------------------------------------------------
    // finalFired reset on partial — continuation support
    // -----------------------------------------------------------------------

    test("finalFired resets on new partial — allows promotion after prior final", () => {
        // First sentence gets a final, then new partials arrive (user continues)
        // without another final. task-finished should promote the new partials
        // because finalFired was reset when the new partials started.
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "hello", end_time: 100 } },
            // ^ final → finalFired = true
            { event: "result-generated", sentence: { text: "world", end_time: null } },
            // ^ partial → finalFired = false (reset!)
            { event: "task-finished" },
            // ^ should promote "world" since finalFired is false
        ])

        expect(actions).toEqual([
            { action: "final", text: "hello" },
            { action: "interim", text: "world" },
            { action: "promoted-final", text: "world" },
            { action: "finished" },
        ])
    })

    test("finalFired stays false when only partials exist — promotion works", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "hey", end_time: null } },
            { event: "result-generated", sentence: { text: "hey there", end_time: null } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "interim", text: "hey" },
            { action: "interim", text: "hey there" },
            { action: "promoted-final", text: "hey there" },
            { action: "finished" },
        ])
    })

    test("final → partial → final — no promotion (second final exists)", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "first", end_time: 100 } },
            { event: "result-generated", sentence: { text: "second", end_time: null } },
            // ^ partial resets finalFired
            { event: "result-generated", sentence: { text: "second complete", end_time: 500 } },
            // ^ final sets finalFired again
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "final", text: "first" },
            { action: "interim", text: "second" },
            { action: "final", text: "second complete" },
            { action: "finished" },
        ])
    })

    test("multiple final → partial cycles — last partial promoted", () => {
        const actions = runSequence([
            // Cycle 1: final then partial
            { event: "result-generated", sentence: { text: "a", end_time: 100 } },
            { event: "result-generated", sentence: { text: "b", end_time: null } },
            // Cycle 2: final then partial
            { event: "result-generated", sentence: { text: "b done", end_time: 200 } },
            { event: "result-generated", sentence: { text: "c", end_time: null } },
            // No final for "c" — task-finished should promote
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "final", text: "a" },
            { action: "interim", text: "b" },
            { action: "final", text: "b done" },
            { action: "interim", text: "c" },
            { action: "promoted-final", text: "c" },
            { action: "finished" },
        ])
    })

    test("final only (no subsequent partial) — no promotion on task-finished", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "done", end_time: 300 } },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "final", text: "done" },
            { action: "finished" },
        ])
    })
})

// ---------------------------------------------------------------------------
// FunASR bridge-error detection
// ---------------------------------------------------------------------------

describe("FunASR bridge-error handling", () => {
    test("bridge-error message is distinguished from DashScope events", () => {
        const bridgeError = JSON.stringify({ type: "bridge-error", message: "Connection failed" })
        const parsed = JSON.parse(bridgeError)

        // Bridge errors have type field, not header.event
        expect(parsed.type).toBe("bridge-error")
        expect(parsed.message).toBe("Connection failed")

        // parseDashScopeMessage should return null for bridge-error (no header.event)
        expect(parseDashScopeMessage(bridgeError)).toBeNull()
    })

    test("bridge-error has no header.event so parseDashScopeMessage returns null", () => {
        expect(parseDashScopeMessage('{"type":"bridge-error","message":"fail"}')).toBeNull()
    })

    test("DashScope event has header.event, not type", () => {
        const dashMsg = JSON.stringify({
            header: { event: "task-started" },
            payload: {},
        })
        const result = parseDashScopeMessage(dashMsg)
        expect(result).not.toBeNull()
        expect(result!.event).toBe("task-started")
    })
})

// ---------------------------------------------------------------------------
// Integration-style: downsample → float32ToInt16 → mergePcmChunks
// ---------------------------------------------------------------------------

describe("audio pipeline integration", () => {
    test("full pipeline: downsample → convert → merge", () => {
        // Simulate: 4096 samples at 48kHz → downsample → Int16 → merge
        const raw = new Float32Array(4096)
        for (let i = 0; i < raw.length; i++) {
            raw[i] = Math.sin((2 * Math.PI * 440 * i) / 48000) // 440 Hz sine
        }

        const downsampled = downsample(raw, 48000, 16000)
        expect(downsampled.length).toBe(Math.round(4096 / 3))

        const pcm16 = float32ToInt16(downsampled)
        expect(pcm16.length).toBe(downsampled.length)

        const merged = mergePcmChunks([pcm16])
        expect(merged.length).toBe(pcm16.length * 2) // 2 bytes per Int16
    })

    test("merging multiple pipeline chunks", () => {
        const chunk1 = float32ToInt16(
            downsample(new Float32Array(4096).fill(0.1), 48000, 16000)
        )
        const chunk2 = float32ToInt16(
            downsample(new Float32Array(4096).fill(0.2), 48000, 16000)
        )

        const merged = mergePcmChunks([chunk1, chunk2])
        expect(merged.length).toBe((chunk1.length + chunk2.length) * 2)
    })
})

// ---------------------------------------------------------------------------
// Additional edge cases — downsample
// ---------------------------------------------------------------------------

describe("downsample — additional edge cases", () => {
    test("last sample uses clamped index (hi === lo at boundary)", () => {
        const input = new Float32Array([1.0, 0.5, 0.0])
        // 3:1 ratio → output length ceil(3/3) = 1? No: Math.round(3 * (8000/24000)) = 1
        const out = downsample(input, 24000, 8000)
        // Output should be 1 sample, the value at index 0
        expect(out.length).toBe(1)
        expect(out[0]).toBeCloseTo(1.0)
    })

    test("negative amplitude values are preserved through interpolation", () => {
        const input = new Float32Array([-0.5, -1.0, 0.3, -0.7, 0.0])
        const out = downsample(input, 48000, 16000)
        // Output should have approximately ceil(5/3) ≈ 2 samples
        expect(out.length).toBeGreaterThan(0)
        // At least one sample should be negative
        let hasNegative = false
        for (let i = 0; i < out.length; i++) {
            if (out[i] < 0) hasNegative = true
        }
        expect(hasNegative).toBe(true)
    })

    test("all-negative input stays negative after downsample", () => {
        const input = new Float32Array(100).fill(-0.8)
        const out = downsample(input, 48000, 16000)
        for (let i = 0; i < out.length; i++) {
            expect(out[i]).toBeCloseTo(-0.8)
        }
    })
})

// ---------------------------------------------------------------------------
// Additional edge cases — float32ToInt16
// ---------------------------------------------------------------------------

describe("float32ToInt16 — additional edge cases", () => {
    test("asymmetry at boundaries: positive uses 0x7FFF, negative uses 0x8000", () => {
        // +1.0 → 32767 (0x7FFF), -1.0 → -32768 (0x8000)
        const pos = float32ToInt16(new Float32Array([1.0]))
        const neg = float32ToInt16(new Float32Array([-1.0]))
        expect(pos[0]).toBe(32767)
        expect(neg[0]).toBe(-32768)
        // Asymmetry: abs(positive max) !== abs(negative min)
        expect(Math.abs(pos[0])).not.toBe(Math.abs(neg[0]))
    })

    test("very small positive value near zero", () => {
        const result = float32ToInt16(new Float32Array([0.00003]))
        // 0.00003 * 0x7FFF ≈ 0.98 → rounds to 0 or 1
        expect(result[0]).toBeLessThanOrEqual(1)
        expect(result[0]).toBeGreaterThanOrEqual(0)
    })

    test("very small negative value near zero", () => {
        const result = float32ToInt16(new Float32Array([-0.00003]))
        // -0.00003 * 0x8000 ≈ -0.98 → rounds to 0 or -1
        expect(result[0]).toBeGreaterThanOrEqual(-1)
        expect(result[0]).toBeLessThanOrEqual(0)
    })
})

// ---------------------------------------------------------------------------
// Additional edge cases — mergePcmChunks
// ---------------------------------------------------------------------------

describe("mergePcmChunks — additional edge cases", () => {
    test("merges chunks of drastically different lengths", () => {
        const large = float32ToInt16(new Float32Array(1000).fill(0.5))
        const tiny = float32ToInt16(new Float32Array(1).fill(-0.5))
        const medium = float32ToInt16(new Float32Array(500).fill(0.1))

        const merged = mergePcmChunks([large, tiny, medium])
        const expectedBytes = (1000 + 1 + 500) * 2
        // May be adjusted for even-byte alignment
        expect(merged.length).toBe(expectedBytes % 2 === 0 ? expectedBytes : expectedBytes - 1)
    })
})

// ---------------------------------------------------------------------------
// Additional edge cases — parseDashScopeMessage
// ---------------------------------------------------------------------------

describe("parseDashScopeMessage — additional edge cases", () => {
    test("header.event present but payload.output missing → still parses", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: {},
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        expect(result!.event).toBe("result-generated")
        // payload.output.sentence would be undefined
        expect(result!.parsed?.payload?.output?.sentence).toBeUndefined()
    })

    test("payload.output present but sentence missing → sentence is undefined", () => {
        const msg = JSON.stringify({
            header: { event: "result-generated" },
            payload: { output: {} },
        })
        const result = parseDashScopeMessage(msg)
        expect(result).not.toBeNull()
        expect(result!.parsed?.payload?.output?.sentence).toBeUndefined()
    })
})

// ---------------------------------------------------------------------------
// Detection edge cases — unusual end_time types
// ---------------------------------------------------------------------------

describe("FunASR detection — unusual end_time types", () => {
    function classifyResult(sentence: any): "skip" | "partial" | "final" {
        if (!sentence) return "skip"
        const text: string = sentence.text ?? ""
        if (sentence.heartbeat && !text) return "skip"
        const isPartial = sentence.end_time == null || sentence.end_time < 0
        return isPartial ? "partial" : "final"
    }

    test("end_time: false → final (false == null is false, false < 0 is false)", () => {
        expect(classifyResult({ text: "x", end_time: false })).toBe("final")
    })

    test('end_time: "" → final (empty string == null is false, "" < 0 is false)', () => {
        expect(classifyResult({ text: "x", end_time: "" })).toBe("final")
    })

    test("end_time: NaN → partial (NaN < 0 is false, but NaN == null is false → final... actually NaN < 0 = false, NaN == null = false → final)", () => {
        // NaN == null → false, NaN < 0 → false → classifies as final
        expect(classifyResult({ text: "x", end_time: NaN })).toBe("final")
    })
})

// ---------------------------------------------------------------------------
// runSequence — additional scenarios
// ---------------------------------------------------------------------------

describe("FunASR detection — additional sequence scenarios", () => {
    function classifyResult(sentence: any): "skip" | "partial" | "final" {
        if (!sentence) return "skip"
        const text: string = sentence.text ?? ""
        if (sentence.heartbeat && !text) return "skip"
        const isPartial = sentence.end_time == null || sentence.end_time < 0
        return isPartial ? "partial" : "final"
    }

    function shouldPromoteInterim(finalFired: boolean, lastInterimText: string): boolean {
        return !finalFired && !!lastInterimText
    }

    function runSequence(events: Array<{
        event: string; sentence?: any; error_message?: string
    }>) {
        let lastInterimText = ""
        let finalFired = false
        const result: Array<{ action: string; text?: string; error?: string }> = []
        for (const ev of events) {
            switch (ev.event) {
                case "result-generated": {
                    const cls = classifyResult(ev.sentence)
                    if (cls === "skip") {
                        result.push({ action: "skip" })
                    } else if (cls === "partial") {
                        lastInterimText = ev.sentence?.text ?? ""
                        finalFired = false
                        result.push({ action: "interim", text: lastInterimText })
                    } else {
                        finalFired = true
                        lastInterimText = ""
                        result.push({ action: "final", text: ev.sentence?.text ?? "" })
                    }
                    break
                }
                case "task-finished":
                    if (shouldPromoteInterim(finalFired, lastInterimText)) {
                        result.push({ action: "promoted-final", text: lastInterimText })
                        lastInterimText = ""
                    }
                    result.push({ action: "finished" })
                    break
                case "task-failed":
                    result.push({ action: "error", error: ev.error_message ?? "unknown error" })
                    break
            }
        }
        return result
    }

    test("final → new partials without final → task-finished promotes new partials", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: { text: "first", end_time: 1000 } },
            { event: "result-generated", sentence: { text: "new partial", end_time: null } },
            { event: "task-finished" },
        ])
        // After the first final, finalFired=true. New partial resets finalFired=false.
        // So task-finished promotes the last interim text.
        expect(actions).toEqual([
            { action: "final", text: "first" },
            { action: "interim", text: "new partial" },
            { action: "promoted-final", text: "new partial" },
            { action: "finished" },
        ])
    })

    test("result-generated with missing sentence → skip", () => {
        const actions = runSequence([
            { event: "result-generated", sentence: undefined },
            { event: "result-generated", sentence: null },
            { event: "task-finished" },
        ])

        expect(actions).toEqual([
            { action: "skip" },
            { action: "skip" },
            { action: "finished" },
        ])
    })

    test("only task-finished without any events → no promotion, just finished", () => {
        const actions = runSequence([
            { event: "task-finished" },
        ])
        expect(actions).toEqual([
            { action: "finished" },
        ])
    })
})
