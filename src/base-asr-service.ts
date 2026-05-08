/**
 * BaseASRService — shared DashScope FunASR bridge protocol.
 *
 * Encapsulates the WebSocket bridge connection, DashScope handshake,
 * message parsing, PCM buffering/flushing, and stop/destroy lifecycle.
 *
 * Subclasses only need to provide:
 *   - `getBridgeUrl()` — the WebSocket bridge URL
 *   - `startAudioCapture()` — platform-specific mic/BLE setup that pushes
 *     Int16 PCM chunks into `this.pcmBuffer`
 *   - `stopAudioCapture()` — release platform audio resources
 *
 * Optional overrides:
 *   - `logPrefix` — log label (default "ASR")
 *   - `validateBeforeStart()` — pre-start guard (e.g. check BLE connection)
 */

import type { ASRCallbacks, IASRService, ASRStartOptions } from "./voice-agent-types"
import {
    ASR_SPEECH_NOISE_THRESHOLD,
    ASR_LANGUAGE_HINTS,
    ASR_CHUNK_INTERVAL_MS,
    ASR_MAX_SENTENCE_SILENCE_MS,
} from "./config"
import { parseDashScopeMessage, mergePcmChunks } from "./audio-helpers"

export abstract class BaseASRService implements IASRService {
    protected callbacks: ASRCallbacks = {}
    protected ws: WebSocket | null = null
    protected isRunning = false
    protected pcmBuffer: Int16Array[] = []
    protected sendTimer: ReturnType<typeof setInterval> | null = null

    private lastInterimText = ""
    private finalFired = false
    private stopPromise: Promise<void> | null = null

    // -----------------------------------------------------------------------
    // Abstract — subclasses must implement
    // -----------------------------------------------------------------------

    /**
     * Start platform-specific audio capture.
     * Push Int16 PCM chunks into `this.pcmBuffer` — the base class
     * flushes them to the WebSocket on a timer.
     */
    protected abstract startAudioCapture(): Promise<void>

    /** Stop platform-specific audio capture and release resources. */
    protected abstract stopAudioCapture(): void

    // -----------------------------------------------------------------------
    // Template methods — override for direct DashScope mode
    // -----------------------------------------------------------------------

    /**
     * Create the WebSocket connection.
     * Default: connects to the bridge via `getBridgeUrl()`.
     * Override for direct DashScope connection (e.g. mobile with auth headers).
     */
    protected createWebSocket(_apiKey?: string): WebSocket {
        return new WebSocket(this.getBridgeUrl())
    }

    /**
     * WebSocket bridge URL (e.g. "ws://127.0.0.1:9231").
     * Used by the default `createWebSocket()`. Override or make it a no-op
     * if your subclass overrides `createWebSocket()` directly.
     */
    protected getBridgeUrl(): string {
        throw new Error(
            `${this.logPrefix}: getBridgeUrl() not implemented — ` +
            "override createWebSocket() for direct mode or implement getBridgeUrl() for bridge mode"
        )
    }

    /**
     * Build the JSON payload to start an ASR task.
     * Default: bridge protocol `{ type: "start", apiKey, ... }`.
     * Override to send DashScope `run-task` directly.
     */
    protected buildStartPayload(
        apiKey: string,
        opts: { maxSilenceMs: number; speechNoiseThreshold: number | null; languageHints: readonly string[] }
    ): string {
        return JSON.stringify({
            type: "start",
            apiKey,
            maxSentenceSilence: opts.maxSilenceMs,
            speechNoiseThreshold: opts.speechNoiseThreshold,
            languageHints: opts.languageHints,
        })
    }

    /**
     * Build the JSON payload to stop the current ASR task.
     * Default: bridge protocol `{ type: "stop" }`.
     * Override to send DashScope `finish-task` directly.
     */
    protected buildStopPayload(): string {
        return JSON.stringify({ type: "stop" })
    }

    // -----------------------------------------------------------------------
    // Optional hooks
    // -----------------------------------------------------------------------

    /** Log prefix for messages (default "ASR"). Override for e.g. "FunASR". */
    protected get logPrefix(): string {
        return "ASR"
    }

    /**
     * Called before start(). Throw to abort (e.g. if no device connected).
     * Default implementation does nothing.
     */
    protected validateBeforeStart(): void {}

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    on(cb: ASRCallbacks): void {
        this.callbacks = { ...this.callbacks, ...cb }
    }

    /**
     * Pre-connect to the bridge WebSocket server.
     * Useful for verifying connectivity during init without starting ASR.
     */
    async connect(): Promise<void> {
        await this.ensureConnection()
    }

    async start(apiKey: string, opts: ASRStartOptions = {}): Promise<void> {
        const {
            maxSilenceMs = ASR_MAX_SENTENCE_SILENCE_MS,
            speechNoiseThreshold = ASR_SPEECH_NOISE_THRESHOLD,
            languageHints = ASR_LANGUAGE_HINTS,
        } = opts

        if (this.isRunning) return
        // Wait for any in-progress stop to finish before reusing the bridge WS
        if (this.stopPromise) await this.stopPromise

        this.validateBeforeStart()

        this.lastInterimText = ""
        this.finalFired = false

        // 1. Ensure WebSocket connection
        await this.ensureConnection(apiKey)

        // 2. Send start command and wait for task-started from DashScope
        await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error("Timed out waiting for DashScope task-started"))
            }, 15_000)

            this.ws!.onmessage = (ev) => {
                if (typeof ev.data !== "string") return
                const msg = parseDashScopeMessage(ev.data)
                if (!msg) {
                    try {
                        const bridgeMsg = JSON.parse(ev.data)
                        if (bridgeMsg.type === "bridge-error") {
                            clearTimeout(timeout)
                            reject(new Error(`Bridge error: ${bridgeMsg.message}`))
                            return
                        }
                    } catch { /* ignore */ }
                    return
                }

                if (msg.event === "task-started") {
                    clearTimeout(timeout)
                    this.callbacks.onLog?.(
                        `${this.logPrefix}: task-started, capturing audio…`
                    )
                    // Switch to streaming message handler
                    this.ws!.onmessage = (e) => this.handleMessage(e)
                    resolve()
                } else if (msg.event === "task-failed") {
                    clearTimeout(timeout)
                    const errMsg =
                        msg.parsed?.header?.error_message ?? "unknown error"
                    reject(
                        new Error(`${this.logPrefix} task-failed: ${errMsg}`)
                    )
                }
            }

            this.ws!.send(
                this.buildStartPayload(apiKey, {
                    maxSilenceMs,
                    speechNoiseThreshold,
                    languageHints,
                })
            )
        })

        // Mark running before audio capture so stop() can clean up on failure
        this.isRunning = true

        // 3. Start platform-specific audio capture
        try {
            await this.startAudioCapture()
        } catch (e: unknown) {
            this.callbacks.onError?.(
                e instanceof Error ? e : new Error(String(e))
            )
            await this.stop()
            return
        }

        // 4. Periodically flush buffered PCM over WebSocket
        this.sendTimer = setInterval(
            () => this.flushAudio(),
            ASR_CHUNK_INTERVAL_MS
        )

        this.callbacks.onLog?.(`${this.logPrefix}: listening…`)
    }

    async stop(): Promise<void> {
        if (!this.isRunning) return
        this.isRunning = false

        this.stopPromise = this.performStop()
        await this.stopPromise
    }

    destroy(): void {
        this.isRunning = false
        this.stopAudioCapture()
        if (this.sendTimer) {
            clearInterval(this.sendTimer)
            this.sendTimer = null
        }
        this.pcmBuffer = []
        this.closeConnection()
    }

    // -----------------------------------------------------------------------
    // Internal — WebSocket connection
    // -----------------------------------------------------------------------

    protected async ensureConnection(apiKey?: string): Promise<void> {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return

        // Clean up stale socket
        this.cleanupSocket()

        const MAX_RETRIES = 3
        let lastError: Error | null = null

        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
            if (attempt > 1) {
                const delay = 1000 * attempt
                this.callbacks.onLog?.(
                    `${this.logPrefix}: reconnect attempt ${attempt}/${MAX_RETRIES} in ${delay}ms…`
                )
                await new Promise(r => setTimeout(r, delay))
            }

            try {
                await this.connectOnce(apiKey)
                return // success
            } catch (err) {
                lastError = err instanceof Error ? err : new Error(String(err))
                this.callbacks.onLog?.(
                    `${this.logPrefix}: connect attempt ${attempt}/${MAX_RETRIES} failed: ${lastError.message}`
                )
                this.cleanupSocket()
            }
        }

        throw lastError ?? new Error(`${this.logPrefix}: connection failed after retries`)
    }

    private cleanupSocket(): void {
        if (this.ws) {
            this.ws.onmessage = null
            this.ws.onerror = null
            this.ws.onclose = null
            try {
                this.ws.close()
            } catch {
                /* ignore */
            }
            this.ws = null
        }
    }

    private connectOnce(apiKey?: string): Promise<void> {
        this.ws = this.createWebSocket(apiKey)
        this.ws.binaryType = "arraybuffer"

        return new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
                try {
                    this.ws?.close()
                } catch {
                    /* ignore */
                }
                reject(
                    new Error(
                        `${this.logPrefix}: WebSocket connection timeout`
                    )
                )
            }, 10_000)

            this.ws!.onopen = () => {
                clearTimeout(timeout)
                this.callbacks.onLog?.(
                    `${this.logPrefix}: WebSocket connected`
                )
                resolve()
            }
            this.ws!.onerror = () => {
                clearTimeout(timeout)
                reject(
                    new Error(
                        `${this.logPrefix}: WebSocket connection failed`
                    )
                )
            }
            this.ws!.onclose = () => {
                this.ws = null
                this.callbacks.onLog?.(
                    `${this.logPrefix}: WebSocket connection closed`
                )
            }
        })
    }

    // -----------------------------------------------------------------------
    // Internal — protocol
    // -----------------------------------------------------------------------

    /**
     * Performs the actual stop sequence: releases audio, flushes remaining
     * PCM, sends the stop command, and waits for task-finished.
     */
    private async performStop(): Promise<void> {
        try {
            // Release platform audio resources first
            this.stopAudioCapture()

            if (this.sendTimer) {
                clearInterval(this.sendTimer)
                this.sendTimer = null
            }

            // Flush any remaining buffered audio
            this.flushAudio()

            // Send stop command
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(this.buildStopPayload())

                // Wait briefly for task-finished acknowledgement
                await new Promise<void>((resolve) => {
                    const timer = setTimeout(resolve, 2_000)
                    const prevHandler = this.ws!.onmessage
                    this.ws!.onmessage = (ev) => {
                        if (prevHandler) prevHandler.call(this.ws!, ev)
                        if (typeof ev.data !== "string") return
                        const msg = parseDashScopeMessage(ev.data)
                        if (
                            msg &&
                            (msg.event === "task-finished" ||
                                msg.event === "task-failed")
                        ) {
                            clearTimeout(timer)
                            resolve()
                        }
                    }
                })
            }

            this.pcmBuffer = []
        } finally {
            this.stopPromise = null
        }
    }

    protected flushAudio(): void {
        if (this.pcmBuffer.length === 0) return
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            // Can't send — discard buffered audio to avoid memory growth
            this.pcmBuffer = []
            return
        }

        const bytes = mergePcmChunks(this.pcmBuffer)
        this.pcmBuffer = []
        if (bytes.length === 0) return

        this.ws.send(bytes)
    }

    private handleMessage(ev: MessageEvent): void {
        if (typeof ev.data !== "string") return

        let parsed: any
        try {
            parsed = JSON.parse(ev.data)
        } catch {
            return
        }

        // Check for bridge-level errors first
        if (parsed?.type === "bridge-error") {
            this.callbacks.onError?.(
                new Error(`Bridge error: ${parsed.message}`)
            )
            return
        }

        // Extract DashScope event
        const event = parsed?.header?.event
        if (!event) return

        this.callbacks.onLog?.(`${this.logPrefix} event: ${event}`)

        switch (event) {
            case "result-generated": {
                const sentence = parsed?.payload?.output?.sentence
                if (!sentence) break
                const text = sentence.text ?? ""
                if (sentence.heartbeat && !text) break
                const isPartial =
                    sentence.end_time == null || sentence.end_time < 0
                if (isPartial) {
                    this.lastInterimText = text
                    this.finalFired = false
                    this.callbacks.onInterim?.(text)
                } else {
                    this.finalFired = true
                    this.lastInterimText = ""
                    this.callbacks.onFinal?.(text)
                }
                break
            }
            case "task-finished":
                if (!this.finalFired && this.lastInterimText) {
                    this.callbacks.onLog?.(
                        `${this.logPrefix}: promoting last interim to final: "${this.lastInterimText}"`
                    )
                    this.callbacks.onFinal?.(this.lastInterimText)
                    this.lastInterimText = ""
                }
                // If isRunning is still true, DashScope ended the task
                // on its own (e.g. server-side silence/inactivity timeout).
                // Clean up so start() can be called again.
                if (this.isRunning) {
                    this.isRunning = false
                    this.stopAudioCapture()
                    if (this.sendTimer) {
                        clearInterval(this.sendTimer)
                        this.sendTimer = null
                    }
                    this.pcmBuffer = []
                }
                this.callbacks.onFinished?.()
                break
            case "task-failed": {
                const errMsg =
                    parsed?.header?.error_message ?? "unknown error"
                // Clean up so the service is in a restartable state.
                // Without this, isRunning stays true and start() no-ops.
                this.isRunning = false
                this.stopAudioCapture()
                if (this.sendTimer) {
                    clearInterval(this.sendTimer)
                    this.sendTimer = null
                }
                this.pcmBuffer = []
                this.callbacks.onError?.(
                    new Error(`${this.logPrefix} error: ${errMsg}`)
                )
                break
            }
        }
    }

    protected closeConnection(): void {
        this.cleanupSocket()
    }
}
