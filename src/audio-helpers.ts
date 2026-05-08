/**
 * Audio processing helper functions — shared across platforms.
 *
 * These are pure functions with no platform dependencies.
 */

/** Safely coerce an unknown caught value to an Error. */
export function toError(e: unknown): Error {
    return e instanceof Error ? e : new Error(String(e))
}

/**
 * Simple linear-interpolation downsampler.
 * Good enough for speech; avoids pulling in a resampling library.
 */
export function downsample(
    input: Float32Array,
    fromRate: number,
    toRate: number
): Float32Array {
    if (fromRate === toRate) return input
    const ratio = fromRate / toRate
    const outputLen = Math.round(input.length / ratio)
    const output = new Float32Array(outputLen)
    for (let i = 0; i < outputLen; i++) {
        const srcIdx = i * ratio
        const lo = Math.floor(srcIdx)
        const hi = Math.min(lo + 1, input.length - 1)
        const frac = srcIdx - lo
        output[i] = input[lo] * (1 - frac) + input[hi] * frac
    }
    return output
}

/** Convert Float32 [-1, 1] samples to Int16 PCM. */
export function float32ToInt16(float32: Float32Array): Int16Array {
    const int16 = new Int16Array(float32.length)
    for (let i = 0; i < float32.length; i++) {
        const s = Math.max(-1, Math.min(1, float32[i]))
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
    }
    return int16
}

/** Merge an array of Int16Array chunks into one contiguous Uint8Array. */
export function mergePcmChunks(chunks: Int16Array[]): Uint8Array<ArrayBuffer> {
    const totalLen = chunks.reduce((s, c) => s + c.length, 0)
    const merged = new Int16Array(totalLen)
    let offset = 0
    for (const chunk of chunks) {
        merged.set(chunk, offset)
        offset += chunk.length
    }
    return new Uint8Array(merged.buffer)
}

/** Parse a DashScope JSON message and extract the event + payload. */
export function parseDashScopeMessage(
    data: string
): { event: string; parsed: any } | null {
    try {
        const parsed = JSON.parse(data)
        const event = parsed?.header?.event
        if (event) return { event, parsed }
    } catch {
        /* ignore */
    }
    return null
}

/**
 * Create a WAV file buffer from Float32 PCM samples.
 * Used for TTS playback audio (Cartesia outputs Float32 at 24 kHz).
 */
export function createWavFromFloat32(
    samples: Float32Array,
    sampleRate: number = 24000
): Uint8Array {
    const int16 = float32ToInt16(samples)
    return createWavFromInt16(int16, sampleRate)
}

/**
 * Create a WAV file buffer from Int16 PCM samples.
 */
export function createWavFromInt16(
    samples: Int16Array,
    sampleRate: number = 16000,
    numChannels: number = 1,
    bitsPerSample: number = 16
): Uint8Array {
    const bytesPerSample = bitsPerSample / 8
    const dataSize = samples.length * bytesPerSample
    const headerSize = 44
    const buffer = new ArrayBuffer(headerSize + dataSize)
    const view = new DataView(buffer)

    // RIFF header
    writeString(view, 0, "RIFF")
    view.setUint32(4, 36 + dataSize, true)
    writeString(view, 8, "WAVE")

    // fmt sub-chunk
    writeString(view, 12, "fmt ")
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true) // PCM format
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * numChannels * bytesPerSample, true)
    view.setUint16(32, numChannels * bytesPerSample, true)
    view.setUint16(34, bitsPerSample, true)

    // data sub-chunk
    writeString(view, 36, "data")
    view.setUint32(40, dataSize, true)

    // Copy PCM samples directly (Int16 LE → WAV LE, no conversion needed)
    new Uint8Array(buffer, headerSize).set(
        new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength)
    )

    return new Uint8Array(buffer)
}

function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i))
    }
}
