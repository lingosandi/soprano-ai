/**
 * LLMClient – generic, browser-compatible streaming client for
 * OpenAI-compatible chat completion endpoints.
 *
 * This is the heart of the shared library.  It handles:
 *   • Building the request payload
 *   • SSE streaming with inactivity / first-chunk timeouts
 *   • Retry with exponential back-off on network and transient HTTP errors
 *   • Pluggable callbacks so the consumer decides how to render output
 *
 * No Node.js-only APIs (chalk, process, fs) are used here – it works in
 * both Node 22+ and modern browsers.
 */

import type {
    FetchResponseLike,
    LLMProviderConfig,
    LLMRequestOptions,
    LLMClientResponse,
    StreamCallbacks
} from "./types"
import { STREAM_INACTIVITY_TIMEOUT, FIRST_CHUNK_TIMEOUT } from "./config"

function shouldOmitTemperature(config: LLMProviderConfig): boolean {
    return config.model === "kimi-k2.5"
}

function buildChatPayload(
    config: LLMProviderConfig,
    basePayload: {
        model: string
        messages: LLMRequestOptions["messages"]
        temperature: number
        max_tokens: number
        frequency_penalty: number
        presence_penalty: number
        stream: boolean
    },
    requestExtraBody?: Record<string, unknown>,
): Record<string, unknown> {
    const payload: Record<string, unknown> = {
        ...basePayload,
        ...(config.extraBody ?? {}),
        ...(requestExtraBody ?? {}),
    }

    // Moonshot's kimi-k2.5 rejects caller-specified temperature values.
    // Omitting the field lets the API apply its own fixed mode-specific value.
    if (shouldOmitTemperature(config)) {
        delete payload.temperature
    }

    return payload
}

// ---------------------------------------------------------------------------
// LLMClient
// ---------------------------------------------------------------------------

export class LLMClient {
    private config: LLMProviderConfig

    constructor(config: LLMProviderConfig) {
        this.config = config
    }

    /** Swap the underlying provider at runtime. */
    setConfig(config: LLMProviderConfig): void {
        this.config = config
    }

    /** Read-only access to the current provider config. */
    getConfig(): LLMProviderConfig {
        return this.config
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * Send a chat completion request.
     *
     * When `options.stream` is true (the default) the response is streamed
     * token-by-token through the supplied `callbacks`.
     */
    async chat(
        options: LLMRequestOptions,
        callbacks?: StreamCallbacks
    ): Promise<LLMClientResponse> {
        const {
            messages,
            temperature = 0.2,
            maxTokens = 16384 * 2,
            stream = true,
            frequencyPenalty = 0,
            presencePenalty = 0,
            extraBody,
            signal
        } = options

        const payload = buildChatPayload(this.config, {
            model: this.config.model,
            messages,
            temperature,
            max_tokens: maxTokens,
            frequency_penalty: frequencyPenalty,
            presence_penalty: presencePenalty,
            stream
        }, extraBody)
        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            Authorization: `Bearer ${this.config.apiKey}`,
            ...(this.config.extraHeaders ?? {})
        }

        const url = `${this.config.baseUrl}/chat/completions`
        const log = callbacks?.onLog ?? (() => {})

        let lastError: unknown
        const maxRetries = 3

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (signal?.aborted) {
                    throw createAbortError(signal.reason)
                }

                if (attempt > 1) {
                    log(`Retry attempt ${attempt}/${maxRetries}…`, "warn")
                    await sleep(1000 * attempt)
                }

                const controller = new AbortController()
                const abortFetch = () => {
                    controller.abort(signal?.reason)
                }
                signal?.addEventListener("abort", abortFetch, { once: true })
                const doFetch = this.config.fetchImpl ?? globalThis.fetch
                try {
                    const response = await doFetch(url, {
                        method: "POST",
                        headers,
                        body: JSON.stringify(payload),
                        signal: controller.signal
                    })

                    if (!response.ok) {
                        const errorText = await response.text()
                        const err = new ApiError(response.status, errorText)
                        // Retry on transient HTTP errors (502, 503, 429, etc.)
                        if (isRetryableStatus(response.status) && attempt < maxRetries) {
                            log(`Transient HTTP ${response.status}. Retrying…`, "warn")
                            lastError = err
                            continue
                        }
                        throw err
                    }

                    if (stream) {
                        return await this.handleStream(
                            response,
                            controller,
                            callbacks,
                            signal
                        )
                    }

                    if (signal?.aborted) {
                        throw createAbortError(signal.reason)
                    }

                    // Non-streaming path
                    const data = (await response.json()) as any
                    const content =
                        data.choices?.[0]?.message?.content ?? ""
                    return {
                        content,
                        finishReason:
                            data.choices?.[0]?.finish_reason ?? "stop",
                        usage: data.usage
                            ? {
                                  promptTokens:
                                      data.usage.prompt_tokens ?? 0,
                                  completionTokens:
                                      data.usage.completion_tokens ?? 0
                              }
                            : undefined
                    }
                } finally {
                    signal?.removeEventListener("abort", abortFetch)
                }
            } catch (err: any) {
                lastError = err
                if (isAbortError(err) && signal?.aborted) {
                    throw createAbortError(signal.reason)
                }
                // Retry on network-level failures (ECONNREFUSED, ETIMEDOUT, etc.)
                if (isNetworkError(err) && attempt < maxRetries) {
                    log(
                        `Network error: ${err.code ?? err.message}. Retrying…`,
                        "warn"
                    )
                    continue
                }
                throw err
            }
        }

        throw lastError ?? new Error("All retry attempts failed")
    }

    // -----------------------------------------------------------------------
    // Streaming
    // -----------------------------------------------------------------------

    private async handleStream(
        response: FetchResponseLike,
        fetchController: AbortController,
        callbacks?: StreamCallbacks,
        signal?: AbortSignal
    ): Promise<LLMClientResponse> {
        let fullContent = ""
        let finishReason = ""
        let streamFrozen = false

        const log = callbacks?.onLog ?? (() => {})

        const body = response.body as ReadableStream<Uint8Array> | null | undefined
        if (!body) {
            throw new Error(
                "Response body is null – streaming not supported"
            )
        }

        const reader = body.getReader()
        const decoder = new TextDecoder()
        let buffer = ""
        let streamAborted = false
        let firstChunkReceived = false

        // Timer handles (stored as number for browser compat)
        let inactivityTimer: ReturnType<typeof setTimeout> | null = null
        let firstChunkTimer: ReturnType<typeof setTimeout> | null = null

        const abortStream = () => {
            streamAborted = true
            cleanup()
            fetchController.abort(signal?.reason)
        }

        if (signal?.aborted) {
            throw createAbortError(signal.reason)
        }

        signal?.addEventListener("abort", abortStream, { once: true })

        const cleanup = () => {
            if (inactivityTimer) {
                clearTimeout(inactivityTimer)
                inactivityTimer = null
            }
            if (firstChunkTimer) {
                clearTimeout(firstChunkTimer)
                firstChunkTimer = null
            }
        }

        const streamPromise = new Promise<void>((resolve, reject) => {
            // --- first-chunk timeout ---
            firstChunkTimer = setTimeout(() => {
                if (!firstChunkReceived && !streamAborted) {
                    streamAborted = true
                    cleanup()
                    reject(
                        new Error(
                            `FIRST_CHUNK_TIMEOUT: No response received after ${FIRST_CHUNK_TIMEOUT / 1000}s`
                        )
                    )
                    fetchController.abort()
                }
            }, FIRST_CHUNK_TIMEOUT)

            // --- inactivity timeout (reset on every chunk) ---
            const resetInactivityTimer = () => {
                if (inactivityTimer) clearTimeout(inactivityTimer)
                inactivityTimer = setTimeout(() => {
                    if (!streamAborted) {
                        streamAborted = true
                        streamFrozen = true
                        cleanup()
                        log(
                            `Stream frozen: no data for ${STREAM_INACTIVITY_TIMEOUT / 1000}s – treating as truncation`,
                            "warn"
                        )
                        callbacks?.onError?.(
                            new Error(
                                "Stream frozen – treating as truncation"
                            )
                        )
                        fetchController.abort()
                        resolve()
                    }
                }, STREAM_INACTIVITY_TIMEOUT)
            }

            // --- read loop ---
            const pump = async () => {
                try {
                    while (true) {
                        if (streamAborted) break

                        const { done, value } = await reader.read()
                        if (done) break

                        if (!firstChunkReceived) {
                            firstChunkReceived = true
                            if (firstChunkTimer) {
                                clearTimeout(firstChunkTimer)
                                firstChunkTimer = null
                            }
                            resetInactivityTimer()
                            callbacks?.onStreamStart?.()
                        }

                        resetInactivityTimer()

                        buffer += decoder.decode(value, { stream: true })
                        const lines = buffer.split("\n")
                        buffer = lines.pop() || ""

                        for (const line of lines) {
                            if (
                                !line.trim() ||
                                !line.startsWith("data: ")
                            )
                                continue
                            if (line.includes("[DONE]")) {
                                streamAborted = true
                                cleanup()
                                break
                            }

                            try {
                                const parsed = JSON.parse(line.slice(6))
                                const delta = parsed.choices?.[0]?.delta
                                const reasoningToken = delta?.reasoning_content
                                if (reasoningToken) {
                                    callbacks?.onReasoningToken?.(reasoningToken)
                                }
                                const token = delta?.content
                                if (!token && !reasoningToken && delta) {
                                    log(`SSE delta (no content): ${JSON.stringify(delta).slice(0, 200)}`, "warn")
                                }
                                if (token) {
                                    fullContent += token
                                    const stopEarly = callbacks?.onToken?.(token)
                                    if (stopEarly === true) {
                                        streamAborted = true
                                        cleanup()
                                        fetchController.abort()
                                        break
                                    }
                                }
                                if (
                                    parsed.choices?.[0]?.finish_reason
                                ) {
                                    finishReason =
                                        parsed.choices[0].finish_reason
                                }
                            } catch {
                                log(`SSE parse error: ${line.slice(0, 200)}`, "warn")
                            }
                        }

                        if (streamAborted) break
                    }
                    resolve()
                } catch (err: any) {
                    cleanup()
                    if (isAbortError(err) && signal?.aborted) {
                        resolve()
                        return
                    }
                    if (!streamAborted) {
                        streamFrozen = true
                        callbacks?.onError?.(err)
                    }
                    resolve()
                }
            }

            pump()
        })

        try {
            await streamPromise
        } catch (err: any) {
            if (err.message?.includes("FIRST_CHUNK_TIMEOUT")) throw err
            throw err
        } finally {
            cleanup()
            signal?.removeEventListener("abort", abortStream)
        }

        if (signal?.aborted) {
            throw createAbortError(signal.reason)
        }

        await callbacks?.onComplete?.(
            fullContent,
            finishReason || (streamFrozen ? "length" : "stop")
        )

        return {
            content: fullContent,
            finishReason: finishReason || (streamFrozen ? "length" : "stop")
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Custom error class for HTTP-level API errors (preserves status code). */
export class ApiError extends Error {
    constructor(public readonly status: number, body: string) {
        super(`API error ${status}: ${body}`)
        this.name = "ApiError"
    }
}

/** HTTP status codes that indicate a transient/retryable server error. */
function isRetryableStatus(status: number): boolean {
    return status === 429 || status === 500 || status === 502 || status === 503 || status === 408
}

export function isNetworkError(err: any): boolean {
    const code = err.code ?? err.cause?.code ?? ""
    return (
        code === "ENOTFOUND" ||
        code === "ETIMEDOUT" ||
        code === "ECONNREFUSED" ||
        code === "UND_ERR_CONNECT_TIMEOUT" ||
        (typeof err.message === "string" &&
            err.message.includes("fetch failed"))
    )
}

function isAbortError(err: unknown): boolean {
    if (!err || typeof err !== "object") {
        return false
    }

    const name = "name" in err ? err.name : undefined
    const code = "code" in err ? err.code : undefined
    const message = "message" in err ? err.message : undefined

    return name === "AbortError"
        || code === "ABORT_ERR"
        || (typeof message === "string" && message.toLowerCase().includes("aborted"))
}

function createAbortError(reason?: unknown): Error {
    const error = reason instanceof Error
        ? reason
        : new Error(
            typeof reason === "string" && reason.trim().length > 0
                ? reason
                : "Request aborted",
        )

    error.name = "AbortError"
    return error
}

function sleep(ms: number): Promise<void> {
    return new Promise((r) => setTimeout(r, ms))
}
