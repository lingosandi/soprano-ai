/**
 * Additional audio helper tests — covers functions NOT tested in funASRHelpers.test.ts:
 *   • toError()
 *   • createWavFromFloat32()
 *   • createWavFromInt16()
 */
import { describe, expect, test } from "vitest"
import {
    toError,
    createWavFromFloat32,
    createWavFromInt16,
    float32ToInt16
} from "../src/audio-helpers"

// ---------------------------------------------------------------------------
// toError
// ---------------------------------------------------------------------------

describe("toError", () => {
    test("returns the same Error when given an Error", () => {
        const err = new Error("test")
        expect(toError(err)).toBe(err)
    })

    test("wraps a string in an Error", () => {
        const result = toError("something went wrong")
        expect(result).toBeInstanceOf(Error)
        expect(result.message).toBe("something went wrong")
    })

    test("wraps a number in an Error", () => {
        const result = toError(42)
        expect(result).toBeInstanceOf(Error)
        expect(result.message).toBe("42")
    })

    test("wraps null in an Error", () => {
        const result = toError(null)
        expect(result).toBeInstanceOf(Error)
        expect(result.message).toBe("null")
    })

    test("wraps undefined in an Error", () => {
        const result = toError(undefined)
        expect(result).toBeInstanceOf(Error)
        expect(result.message).toBe("undefined")
    })

    test("wraps an object in an Error", () => {
        const result = toError({ code: 500 })
        expect(result).toBeInstanceOf(Error)
        expect(result.message).toContain("object")
    })
})

// ---------------------------------------------------------------------------
// createWavFromInt16
// ---------------------------------------------------------------------------

describe("createWavFromInt16", () => {
    test("produces a buffer starting with RIFF header", () => {
        const samples = new Int16Array([0, 100, -100, 200])
        const wav = createWavFromInt16(samples)
        const riff = String.fromCharCode(wav[0], wav[1], wav[2], wav[3])
        expect(riff).toBe("RIFF")
    })

    test("contains WAVE format marker", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const wave = String.fromCharCode(wav[8], wav[9], wav[10], wav[11])
        expect(wave).toBe("WAVE")
    })

    test("contains fmt sub-chunk", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const fmt = String.fromCharCode(wav[12], wav[13], wav[14], wav[15])
        expect(fmt).toBe("fmt ")
    })

    test("contains data sub-chunk", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const dataLabel = String.fromCharCode(wav[36], wav[37], wav[38], wav[39])
        expect(dataLabel).toBe("data")
    })

    test("total size = 44 header + samples * 2 bytes", () => {
        const samples = new Int16Array([10, 20, 30])
        const wav = createWavFromInt16(samples)
        expect(wav.length).toBe(44 + samples.length * 2)
    })

    test("sample rate is embedded in header (bytes 24-27)", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples, 16000)
        const view = new DataView(wav.buffer)
        expect(view.getUint32(24, true)).toBe(16000)
    })

    test("custom sample rate is stored correctly", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples, 44100)
        const view = new DataView(wav.buffer)
        expect(view.getUint32(24, true)).toBe(44100)
    })

    test("PCM format = 1 (bytes 20-21)", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const view = new DataView(wav.buffer)
        expect(view.getUint16(20, true)).toBe(1)
    })

    test("num channels defaults to 1 (bytes 22-23)", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const view = new DataView(wav.buffer)
        expect(view.getUint16(22, true)).toBe(1)
    })

    test("bits per sample defaults to 16 (bytes 34-35)", () => {
        const samples = new Int16Array([0])
        const wav = createWavFromInt16(samples)
        const view = new DataView(wav.buffer)
        expect(view.getUint16(34, true)).toBe(16)
    })

    test("data chunk size matches actual data", () => {
        const samples = new Int16Array([100, 200, 300, 400, 500])
        const wav = createWavFromInt16(samples)
        const view = new DataView(wav.buffer)
        const dataSize = view.getUint32(40, true)
        expect(dataSize).toBe(samples.length * 2)
    })

    test("RIFF chunk size = file size - 8", () => {
        const samples = new Int16Array([1, 2, 3])
        const wav = createWavFromInt16(samples)
        const view = new DataView(wav.buffer)
        const riffSize = view.getUint32(4, true)
        expect(riffSize).toBe(wav.length - 8)
    })

    test("empty samples produce valid WAV (header only)", () => {
        const samples = new Int16Array([])
        const wav = createWavFromInt16(samples)
        expect(wav.length).toBe(44)
        const riff = String.fromCharCode(wav[0], wav[1], wav[2], wav[3])
        expect(riff).toBe("RIFF")
    })
})

// ---------------------------------------------------------------------------
// createWavFromFloat32
// ---------------------------------------------------------------------------

describe("createWavFromFloat32", () => {
    test("converts Float32 to WAV via Int16 path", () => {
        const samples = new Float32Array([0, 0.5, -0.5, 1.0, -1.0])
        const wav = createWavFromFloat32(samples, 24000)
        expect(wav.length).toBe(44 + samples.length * 2)
    })

    test("default sample rate is 24000", () => {
        const samples = new Float32Array([0])
        const wav = createWavFromFloat32(samples)
        const view = new DataView(wav.buffer)
        expect(view.getUint32(24, true)).toBe(24000)
    })

    test("preserves data integrity through Float32 → Int16 → WAV", () => {
        const input = new Float32Array([0.5])
        const wav = createWavFromFloat32(input, 16000)

        // Extract the Int16 sample from the WAV data section
        const view = new DataView(wav.buffer)
        const sample = view.getInt16(44, true)

        // float32ToInt16 converts 0.5 → 0.5 * 0x7FFF ≈ 16383
        const expected = float32ToInt16(input)[0]
        expect(sample).toBe(expected)
    })

    test("empty Float32 produces header-only WAV", () => {
        const wav = createWavFromFloat32(new Float32Array([]))
        expect(wav.length).toBe(44)
    })
})
