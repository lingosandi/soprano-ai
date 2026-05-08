/**
 * CartesiaTTS — Low-latency streaming TTS via the Cartesia WebSocket API.
 *
 * Uses the WebSocket endpoint directly (no SDK) so we have full control over
 * input streaming with continuations, which is critical for voice-agent
 * latency: LLM tokens are forwarded to Cartesia the moment they arrive and
 * audio chunks are played back immediately.
 *
 * Protocol reference:
 *   wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10&api_key=…
 *
 * Each "context" (identified by `context_id`) is a single turn of speech.
 * Intermediate chunks use `continue: true`; the final chunk uses
 * `continue: false` so Cartesia knows to flush & finalize.
 *
 * This module is platform-agnostic — it only requires a standard WebSocket.
 */

import {
    CARTESIA_WS_URL,
    CARTESIA_VERSION,
    CARTESIA_MODEL_ID,
    CARTESIA_VOICE_ID,
    CARTESIA_OUTPUT_FORMAT,
} from "./config"

import type { CartesiaTTSCallbacks } from "./voice-agent-types"
import { toError } from "./audio-helpers"

// ---------------------------------------------------------------------------
// CartesiaTTS
// ---------------------------------------------------------------------------

export class CartesiaTTS {
    private apiKey: string
    private ws: WebSocket | null = null
    private callbacks: CartesiaTTSCallbacks = {}
    private connectPromise: Promise<void> | null = null
    private isConnected = false

    /**
     * Send queue: serializes all sendChunk / flush / cancel calls so that
     * messages reach Cartesia in strict order.
     */
    private sendQueue: Promise<void> = Promise.resolve()

    /**
     * Contexts we explicitly cancelled. Their error/done responses are
     * suppressed so callers don't see "Invalid context ID".
     */
    private cancelledContexts = new Set<string>()

    /**
     * Pending resolvers for `waitForContextDone()`. Keyed by context_id.
     * Resolved when the "done" event fires for that context.
     */
    private pendingDoneResolvers = new Map<string, () => void>()

    /**
     * Context IDs that have already received a "done" event.
     * Allows `waitForContextDone` to resolve immediately if the event
     * arrived before the call was made.
     */
    private completedContexts = new Set<string>()

    constructor(apiKey: string) {
        this.apiKey = apiKey
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /** Register event callbacks (can be called before or after connect). */
    on(callbacks: CartesiaTTSCallbacks): void {
        this.callbacks = { ...this.callbacks, ...callbacks }
    }

    /** Open the WebSocket connection to Cartesia. Retries up to 3 times with exponential backoff. */
    async connect(): Promise<void> {
        if (this.isConnected && this.ws?.readyState === WebSocket.OPEN) return
        if (this.connectPromise) return this.connectPromise

        this.connectPromise = this.connectWithRetry()
        try {
            await this.connectPromise
        } finally {
            this.connectPromise = null
        }
    }

    private async connectWithRetry(maxRetries = 3): Promise<void> {
        let lastError: Error | null = null

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            if (attempt > 1) {
                const delay = 1000 * attempt
                this.callbacks.onLog?.(`Cartesia reconnect attempt ${attempt}/${maxRetries} in ${delay}ms…`)
                await new Promise(r => setTimeout(r, delay))
            }

            try {
                await this.connectOnce()
                return // success
            } catch (err) {
                lastError = toError(err)
                this.callbacks.onLog?.(`Cartesia connect attempt ${attempt}/${maxRetries} failed: ${lastError.message}`)
                // Clean up failed socket
                if (this.ws) {
                    try { this.ws.close() } catch { /* ignore */ }
                    this.ws = null
                }
                this.isConnected = false
            }
        }

        throw lastError ?? new Error("Cartesia connection failed after retries")
    }

    private connectOnce(): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            const url = `${CARTESIA_WS_URL}?cartesia_version=${CARTESIA_VERSION}&api_key=${this.apiKey}`
            const ws = new WebSocket(url)
            let settled = false

            const timeout = setTimeout(() => {
                if (settled) return
                settled = true
                this.isConnected = false
                ws.close()
                reject(new Error("Cartesia WebSocket connection timeout"))
            }, 15_000)

            ws.onopen = () => {
                if (settled) return
                settled = true
                clearTimeout(timeout)
                this.isConnected = true
                this.callbacks.onLog?.("Cartesia WebSocket connected")
                resolve()
            }

            ws.onerror = () => {
                this.isConnected = false
                if (!settled) {
                    settled = true
                    clearTimeout(timeout)
                    reject(new Error("Cartesia WebSocket error"))
                }
            }

            ws.onclose = (ev) => {
                this.isConnected = false
                if (!settled) {
                    settled = true
                    clearTimeout(timeout)
                    reject(new Error("Cartesia WebSocket closed before open"))
                } else if (this.ws === ws) {
                    // Unexpected close after successful connection —
                    // clean up so ensureConnected() triggers a fresh
                    // connection on the next send attempt.
                    // Guard: only clear if this is still OUR socket
                    // (prevents race if disconnect+reconnect happened).
                    this.ws = null
                    this.callbacks.onLog?.(
                        `Cartesia WebSocket closed unexpectedly (code=${ev.code}) — will reconnect on next use`
                    )
                }
            }

            ws.onmessage = (ev) => this.handleMessage(ev)

            this.ws = ws
        })
    }

    /** Disconnect from Cartesia. */
    disconnect(): void {
        if (this.ws) {
            this.ws.close()
            this.ws = null
        }
        this.isConnected = false
        this.connectPromise = null
        this.sendQueue = Promise.resolve()
        this.cancelledContexts.clear()
        // Resolve all pending done waiters so they don't hang
        for (const resolver of this.pendingDoneResolvers.values()) {
            resolver()
        }
        this.pendingDoneResolvers.clear()
        this.completedContexts.clear()
    }

    /**
     * Send a transcript chunk for a given context.
     * Sends are serialized through a queue to guarantee ordering.
     */
    async sendChunk(
        contextId: string,
        transcript: string,
        isContinuation: boolean
    ): Promise<void> {
        this.sendQueue = this.sendQueue
            .then(() => this.doSendChunk(contextId, transcript, isContinuation))
            .catch((e) => this.callbacks.onError?.(toError(e)))
        return this.sendQueue
    }

    /** Convenience: flush (finalize) a context. */
    async flush(contextId: string): Promise<void> {
        return this.sendChunk(contextId, " ", false)
    }

    /** Cancel an active context (stops any in-progress generation). */
    async cancel(contextId: string): Promise<void> {
        this.cancelledContexts.add(contextId)
        // Resolve any pending done waiter — context won't complete normally
        const resolver = this.pendingDoneResolvers.get(contextId)
        if (resolver) {
            this.pendingDoneResolvers.delete(contextId)
            resolver()
        }
        // Safety: cap the set size to prevent unbounded growth if cancelled
        // contexts never receive a response (e.g. network disconnect).
        if (this.cancelledContexts.size > 50) {
            const oldest = this.cancelledContexts.values().next().value!
            this.cancelledContexts.delete(oldest)
        }
        this.sendQueue = this.sendQueue
            .then(async () => {
                await this.ensureConnected()
                try {
                    this.ws!.send(
                        JSON.stringify({
                            context_id: contextId,
                            cancel: true,
                        })
                    )
                } catch {
                    // Socket died between ensureConnected and send —
                    // mark disconnected so next call retries.
                    this.isConnected = false
                    this.ws = null
                    // Cancel is fire-and-forget; swallow the error.
                }
            })
            .catch((e) => this.callbacks.onError?.(toError(e)))
        return this.sendQueue
    }

    /**
     * Wait for a TTS context to finish generating all audio.
     *
     * Resolves when Cartesia sends the "done" event for the given context,
     * when the context is cancelled, or after a timeout.
     *
     * This is critical to prevent audio interleaving: without waiting for
     * "done", a new TTS context could start while the previous one is
     * still generating audio, causing alternating chunks from both contexts
     * to arrive on the WebSocket and get enqueued to the audio player.
     */
    async waitForContextDone(
        contextId: string,
        timeoutMs = 10_000
    ): Promise<void> {
        // Already completed before we started waiting
        if (this.completedContexts.has(contextId)) {
            this.completedContexts.delete(contextId)
            return
        }
        // Context was cancelled — no "done" will arrive
        if (this.cancelledContexts.has(contextId)) {
            return
        }

        return new Promise<void>((resolve) => {
            const timer = setTimeout(() => {
                this.pendingDoneResolvers.delete(contextId)
                this.callbacks.onLog?.(
                    `TTS context done wait timed out (${timeoutMs / 1000}s) — proceeding`
                )
                resolve()
            }, timeoutMs)

            this.pendingDoneResolvers.set(contextId, () => {
                clearTimeout(timer)
                resolve()
            })
        })
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    private async ensureConnected(): Promise<void> {
        if (!this.isConnected || this.ws?.readyState !== WebSocket.OPEN) {
            await this.connect()
        }
    }

    private async doSendChunk(
        contextId: string,
        transcript: string,
        isContinuation: boolean
    ): Promise<void> {
        await this.ensureConnected()

        const msg = {
            model_id: CARTESIA_MODEL_ID,
            transcript,
            voice: {
                mode: "id",
                id: CARTESIA_VOICE_ID,
            },
            output_format: CARTESIA_OUTPUT_FORMAT,
            context_id: contextId,
            continue: isContinuation,
        }

        try {
            this.ws!.send(JSON.stringify(msg))
        } catch {
            // Socket died between ensureConnected and send — mark
            // disconnected so the next call retries the connection.
            this.isConnected = false
            this.ws = null
            throw new Error("Cartesia WebSocket send failed — will reconnect on next attempt")
        }
    }

    private handleMessage(ev: MessageEvent): void {
        let msg: any
        try {
            const data =
                typeof ev.data === "string" ? ev.data : String(ev.data)
            msg = JSON.parse(data)
        } catch {
            // non-JSON frame — ignore
            return
        }

        try {
            if (msg.type === "chunk") {
                if (this.cancelledContexts.has(msg.context_id)) return

                // Audio data arrives as base64-encoded PCM bytes
                const b64: string = msg.data
                const binaryStr = atob(b64)
                const len = binaryStr.length
                const bytes = new Uint8Array(len)
                for (let i = 0; i < len; i++)
                    bytes[i] = binaryStr.charCodeAt(i)
                // Int16Array view on the raw bytes (LE, matches pcm_s16le)
                const pcm16 = new Int16Array(bytes.buffer, 0, len >> 1)
                // Convert Int16 → Float32 in [-1, 1]
                const float32 = new Float32Array(pcm16.length)
                for (let i = 0; i < pcm16.length; i++) {
                    float32[i] = pcm16[i] / 32768
                }
                this.callbacks.onAudio?.(float32)
            } else if (msg.type === "done") {
                // Track completion for waitForContextDone (before cancelled
                // check — handles races where cancel() runs after waiter
                // is registered but before "done" arrives)
                this.completedContexts.add(msg.context_id)
                if (this.completedContexts.size > 50) {
                    const oldest =
                        this.completedContexts.values().next().value!
                    this.completedContexts.delete(oldest)
                }
                // Resolve any pending done waiter
                const doneResolver = this.pendingDoneResolvers.get(
                    msg.context_id
                )
                if (doneResolver) {
                    this.pendingDoneResolvers.delete(msg.context_id)
                    // Entry is consumed by the waiter — remove from
                    // completedContexts so stale IDs don't accumulate.
                    this.completedContexts.delete(msg.context_id)
                    doneResolver()
                }
                if (this.cancelledContexts.delete(msg.context_id)) return
                this.callbacks.onDone?.(msg.context_id)
            } else if (msg.type === "error") {
                // Resolve any pending done waiter so it doesn't hang
                const errResolver = this.pendingDoneResolvers.get(
                    msg.context_id
                )
                if (errResolver) {
                    this.pendingDoneResolvers.delete(msg.context_id)
                    errResolver()
                }
                if (this.cancelledContexts.delete(msg.context_id)) return
                this.callbacks.onError?.(
                    new Error(
                        `Cartesia error: ${msg.message ?? JSON.stringify(msg)}`
                    )
                )
            }
        } catch (err) {
            // Audio processing error — don't silently swallow
            this.callbacks.onError?.(
                err instanceof Error ? err : new Error(String(err))
            )
        }
    }
}
