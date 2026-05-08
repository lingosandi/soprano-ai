/**
 * VoiceAgentService — orchestrates the streaming voice agent pipeline:
 *
 *   User speech (ASR) ──▶ LLM (Qwen, streaming) ──▶ Cartesia TTS ──▶ Audio playback
 *
 * All three stages run concurrently: LLM tokens are forwarded to Cartesia
 * the instant they arrive, and audio chunks are played as soon as they are
 * decoded. This gives the lowest achievable end-to-end latency.
 *
 * Platform-specific audio playback and speech recognition are injected via
 * the IAudioPlayer and IASRService interfaces, allowing this service to
 * run on web (Tauri), React Native (mobile), or any other platform.
 */

import {
    createQwenPlusProviderConfig,
    VOICE_AGENT_PROVIDER,
    VOICE_LLM_MAX_TOKENS,
    VOICE_LLM_REQUEST_EXTRA_BODY,
    VOICE_LLM_TEMPERATURE,
    type CartesiaTTSQuality,
} from "./config"
import { toError } from "./audio-helpers"
import type {
    ChatMessage,
    FetchLike,
    LLMProviderConfig,
} from "./types"
import type {
    AgentState,
    IAudioPlayer,
    IASRService,
    ASRStartOptions,
    BackgroundToolResult,
} from "./voice-agent-types"

import {
    ASR_MAX_SENTENCE_SILENCE_MS,
    ASR_SPEECH_NOISE_THRESHOLD,
    ASR_LANGUAGE_HINTS,
    LLM_TOKEN_BUFFER_THRESHOLD,
    VOICE_AGENT_SYSTEM_PROMPT,
} from "./config"
import type { ToolRegistry } from "./tool-registry"

import { CartesiaTTS } from "./cartesia-tts"
import { buildToolAwareSystemPrompt } from "./prompt-helpers"
import { ToolCallingAgentBase } from "./tool-calling-agent-base"
import { VoiceTurnController } from "./voice-turn-controller"
import type { VoiceMemoryStorage } from "./voice-memory"
import { formatToolCallAsUnixCommand } from "./unix-tooling"
import { VoiceValidationEnforcer } from "./voice-validation-enforcer"
import { createConversationSink } from "./validation-enforcer"

/** Punctuation marks (English + Chinese) that trigger an early TTS flush. */
const SENTENCE_END_RE = /[.!?,;:\n。！？，；：、]$/

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

export interface VoiceAgentDeps {
    /** Platform-specific audio player implementation. */
    player: IAudioPlayer
    /** Platform-specific ASR implementation. */
    asr: IASRService
    /** Custom fetch for LLM streaming (e.g. expo/fetch on React Native). */
    fetchImpl?: FetchLike
    /** Explicit provider credentials. No API keys are read from the environment. */
    apiKeys: SopranoApiKeys
    /** OpenAI-compatible provider config used for voice turns. No default API key is bundled. */
    llmProviderConfig?: LLMProviderConfig
    /** Optional provider config used for memory compaction. Defaults to llmProviderConfig. */
    compactionProviderConfig?: LLMProviderConfig | null
    /** Override the default spoken system prompt. */
    systemPrompt?: string
    /** Exact greeting line spoken during first-time init(). Returning users skip greeting. */
    greetingMessage?: string
    /** LLM sampling temperature for voice turns. */
    llmTemperature?: number
    /** Max completion tokens per voice turn. */
    llmMaxTokens?: number
    /** Provider-specific request fields merged into each voice LLM request. */
    llmRequestExtraBody?: Record<string, unknown>
    /** Cartesia PCM quality for the default TTS instance. Defaults to low. */
    ttsQuality?: CartesiaTTSQuality
    /** Cartesia voice ID for the default TTS instance. Defaults to the package voice. */
    ttsVoiceId?: string
    /** Pre-created TTS implementation, useful for tests or custom Cartesia settings. */
    tts?: CartesiaTTS
    /** Platform-specific ASR tuning overrides (e.g. longer silence for BLE mics). */
    asrOptions?: Partial<ASRStartOptions>
    /** Foreground voice-turn tool registry. */
    toolRegistry?: ToolRegistry
    /**
     * Optional persistent memory storage adapter.
     * When provided, conversation history is persisted across sessions with
    * episodic compaction (inspired by Hammer's memory layer).
     */
    memory?: VoiceMemoryStorage
    /**
     * Optional poller for background task completions.
     * When provided, the service polls periodically and announces completed
     * tasks via the LLM → TTS pipeline. The function should return
     * completed tasks and clear them from the server (drain semantics).
     */
    backgroundTaskPoller?: () => Promise<BackgroundToolResult[]>
}

export class VoiceAgentService extends ToolCallingAgentBase {
    private tts: CartesiaTTS
    private player: IAudioPlayer
    private asr: IASRService
    private conversationHistory: ChatMessage[] = []
    private contextCounter = 0
    private currentContextId: string | null = null
    private pipelineGeneration = 0
    private ttsAudioStarted = false
    private continuationPending = false
    /** True once ASR is stopped for the current playback turn. Reset on startListening / new pipeline. */
    private asrStoppedForPlayback = false
    /** Explicit microphone mute state, independent of transient ASR internals. */
    private micMuted = false

    // Abort in-flight requests
    private abortController: AbortController | null = null

    // Audio level polling
    private audioLevelTimer: ReturnType<typeof setInterval> | null = null

    /** Platform-specific ASR tuning overrides. */
    private asrOverrides: Partial<ASRStartOptions>

    private readonly asrApiKey: string
    private readonly systemPrompt: string
    private readonly greetingMessage: string | null
    private readonly llmTemperature: number
    private readonly llmMaxTokens: number
    private readonly llmRequestExtraBody: Record<string, unknown>

    /** Queue of completed background tasks awaiting spoken announcement. */
    private pendingAnnouncements: BackgroundToolResult[] = []

    /** Polling timer for background task completion checks. */
    private taskPollingTimer: ReturnType<typeof setInterval> | null = null

    /** Injected function to poll for completed background tasks. */
    private backgroundTaskPoller?: () => Promise<BackgroundToolResult[]>

    /** Polling interval in ms for background task completion checks. */
    private static readonly TASK_POLL_INTERVAL_MS = 1000

    /** Whether the lightweight text-only init has been performed. */
    private textReadyPromise: Promise<void> | null = null

    /** Shared foreground reasoning / tool orchestration controller. */
    private turnController: VoiceTurnController

    constructor(deps: VoiceAgentDeps) {
        const llmProviderConfig = deps.llmProviderConfig ?? createQwenPlusProviderConfig({
            apiKey: deps.apiKeys.qwenApiKey,
        })

        super({
            provider: VOICE_AGENT_PROVIDER,
            llmProviderConfig,
            compactionProviderConfig: deps.compactionProviderConfig ?? llmProviderConfig,
            fetchImpl: deps.fetchImpl,
            toolRegistry: deps.toolRegistry,
            memory: deps.memory ?? null,
        })

        // TTS: Cartesia WebSocket
        this.tts = deps.tts ?? new CartesiaTTS(deps.apiKeys.cartesiaApiKey, {
            quality: deps.ttsQuality,
            voiceId: deps.ttsVoiceId,
        })

        // Platform-specific injected dependencies
        this.player = deps.player
        this.asr = deps.asr
        this.micMuted = deps.asr.isMicMuted ?? false
        this.asrOverrides = deps.asrOptions ?? {}
        this.asrApiKey = deps.apiKeys.asrApiKey ?? deps.apiKeys.qwenApiKey
        this.systemPrompt = deps.systemPrompt ?? VOICE_AGENT_SYSTEM_PROMPT
        this.greetingMessage = deps.greetingMessage?.trim() || null
        this.llmTemperature = deps.llmTemperature ?? VOICE_LLM_TEMPERATURE
        this.llmMaxTokens = deps.llmMaxTokens ?? VOICE_LLM_MAX_TOKENS
        this.llmRequestExtraBody = deps.llmRequestExtraBody ?? VOICE_LLM_REQUEST_EXTRA_BODY

        this.backgroundTaskPoller = deps.backgroundTaskPoller

        // Validation enforcer — uses memory.appendMessage when available,
        // otherwise falls back to pushing onto conversationHistory[].
        // Memory is created asynchronously in init(), so the sink checks
        // this.memory lazily rather than capturing it at construction time.
        const sink = createConversationSink((role, content) => {
            if (this.memory) {
                this.memory.appendMessage(role, content)
            } else {
                this.conversationHistory.push({ role, content })
            }
        })
        const enforcer = new VoiceValidationEnforcer(
            sink,
            (msg) => this.callbacks.onLog?.(msg),
        )

        this.turnController = new VoiceTurnController({
            runtime: this.runtime,
            enforcer,
            getFallbackHistory: () => this.conversationHistory,
            getSystemPrompt: () => this.buildSystemPrompt(),
            temperature: this.llmTemperature,
            maxTokens: this.llmMaxTokens,
            requestExtraBody: this.llmRequestExtraBody,
            allowedRunTargets: this.runtime.toolRegistry.getAvailableRunTargets?.() ?? [],
            speechBufferThreshold: LLM_TOKEN_BUFFER_THRESHOLD,
            sentenceEndPattern: SENTENCE_END_RE,
            sendSpeechChunk: (contextId, chunk) =>
                this.tts.sendChunk(contextId, chunk, true),
            flushSpeech: (contextId) => this.tts.flush(contextId),
            onStreamStart: () => {
                this.callbacks.onLog?.("LLM stream started")
            },
            onFirstSpeechToken: ({ skipSpeech }) => {
                this.ttsAudioStarted = true
                if (!skipSpeech) {
                    this.setState("speaking")
                    this.callbacks.onLog?.("First speech token sent to TTS")
                }
            },
            onPartialResponse: (content) => {
                this.callbacks.onPartialResponse?.(content)
            },
            onToolCall: (selectedToolCall) => {
                this.callbacks.onToolCall?.(selectedToolCall)
                this.ttsAudioStarted = true
            },
            onToolCallError: (error, rawCommandText) => {
                this.callbacks.onToolCallError?.(error, rawCommandText)
            },
            onFullResponse: (content) => {
                this.callbacks.onFullResponse?.(content)
            },
            onStreamError: async (error) => {
                this.callbacks.onError?.(error)
                try {
                    await enforcer.handleStreamError(error)
                } finally {
                    this.restoreListeningOrIdle()
                }
            },
            abortCurrentTurn: () => {
                this.abortController?.abort()
                this.currentContextId = null
            },
            onLog: (message) => this.callbacks.onLog?.(message),
            onError: (error) => this.callbacks.onError?.(error),
            appendToolMessage: async (content) => {
                await this.runtime.appendHistoryMessage({
                    role: "tool",
                    content,
                    reason: "Voice tool result save failed",
                    fallbackHistory: this.conversationHistory,
                })
            },
            onToolStart: (name, parameters) => {
                const displayName = formatToolCallAsUnixCommand({
                    name,
                    parameters,
                }, this.getRegisteredToolDefinitions()) ?? name
                this.callbacks.onLog?.(
                    `Running tool: ${displayName}`,
                )
            },
            afterToolExecution: async ({
                runOptions,
                loopOptions,
                turnResult,
            }) => {
                if (turnResult.speechSent && !loopOptions.skipSpeech) {
                    await this.tts.waitForContextDone(loopOptions.contextId)
                    await this.player.waitForEnd()
                }

                return { aborted: runOptions.signal.aborted }
            },
            beforeTurn: ({ isContinuation }) => {
                if (isContinuation) {
                    this.callbacks.onLog?.(
                        "Re-calling LLM for tool-result summary",
                    )
                }

                this.prepareVoiceTurn()
            },
            createTurnOptions: ({
                runOptions,
                isContinuation,
                previousOptions,
            }) =>
                this.createVoiceTurnOptions(
                    isContinuation ? "" : runOptions.text,
                    isContinuation ? true : runOptions.skipHistoryPush,
                    previousOptions?.skipSpeech ?? runOptions.skipSpeech,
                    runOptions.signal,
                ),
            finalizeResult: async ({ runOptions, finalTurnResult }) => {
                this.callbacks.onLog?.("LLM chat() resolved")

                if (finalTurnResult.aborted || runOptions.signal.aborted) {
                    return
                }

                this.currentContextId = null

                if (runOptions.skipSpeech) {
                    if (this.pendingAnnouncements.length > 0) {
                        this.drainAndAnnounceCompletions()
                        return
                    }

                    this.setState("idle")
                    return
                }

                await this.player.waitForEnd()
                if (runOptions.signal.aborted) {
                    return
                }

                if (this.pendingAnnouncements.length > 0) {
                    this.drainAndAnnounceCompletions()
                    return
                }

                if (this.continuousMode && !this.isMicMuted) {
                    this.callbacks.onLog?.("Playback done — restarting mic")
                    this.setState("idle")
                    this.startListening(true).catch((e) =>
                        this.callbacks.onError?.(toError(e)),
                    )
                } else {
                    this.setState("idle")
                }
            },
            restoreAfterError: async () => {
                this.restoreListeningOrIdle()
            },
        })
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /** Get the current TTS output audio level (0–1). */
    getAudioLevel(): number {
        return this.player.getAudioLevel()
    }

    /**
     * Mute/unmute the microphone.
     * Also stops the ASR DashScope session on mute (preventing the ~23 s
     * server-side timeout) and restarts it on unmute if still listening.
     */
    setMicMuted(muted: boolean): void {
        const wasCapturingAudio = this.shouldCaptureAudioForASR()
        this.micMuted = muted
        const isCapturingAudio = this.shouldCaptureAudioForASR()

        this.syncASRMuteState()

        if (wasCapturingAudio && !isCapturingAudio) {
            // End the DashScope task so it doesn't time out after ~23 s
            // of silence. The WebSocket connection stays open for fast
            // restart on unmute.
            this.stopASRSession()
        } else if (
            !wasCapturingAudio &&
            isCapturingAudio &&
            this.shouldRestartASRForCurrentState()
        ) {
            // Re-establish the ASR session now that audio will flow again.
            if (this.state === "listening") {
                this.setState("connecting")
            }
            this.startASRWithRetry().catch((err) => {
                this.callbacks.onLog?.(
                    `ASR restart after unmute failed: ${toError(err).message}`
                )
                this.callbacks.onError?.(toError(err))
                if (this.state === "connecting") {
                    this.setState("idle")
                }
            })
        }
    }

    /** Whether the microphone is currently muted. */
    get isMicMuted(): boolean {
        return this.micMuted
    }

    private shouldCaptureAudioForASR(): boolean {
        return !this.micMuted
    }

    private shouldMaintainASRSession(): boolean {
        return this.continuousMode && this.shouldCaptureAudioForASR()
    }

    private shouldRestartASRForCurrentState(): boolean {
        return this.shouldMaintainASRSession() && this.state === "listening"
    }

    private canRetryASRForCurrentState(): boolean {
        return this.state === "listening" || this.state === "connecting"
    }

    private syncASRMuteState(): void {
        this.asr.setMicMuted?.(this.micMuted)
    }

    private stopASRSession(): void {
        this.asr.stop().catch(() => {})
    }

    private async stopASRSessionAwaited(): Promise<void> {
        await this.asr.stop()
    }

    private destroyASRSession(): void {
        this.asr.destroy()
    }

    /** Start polling audio levels. Uses setInterval for cross-platform compat. */
    private startAudioLevelPolling(): void {
        if (this.audioLevelTimer !== null) return
        this.audioLevelTimer = setInterval(() => {
            this.callbacks.onAudioLevel?.(this.getAudioLevel())
        }, 1000 / 60) // ~60 fps
    }

    /** Stop polling audio levels. */
    private stopAudioLevelPolling(): void {
        if (this.audioLevelTimer !== null) {
            clearInterval(this.audioLevelTimer)
            this.audioLevelTimer = null
        }
    }

    /** Initialize audio + Cartesia + ASR. Call from a user gesture. */
    async init(): Promise<void> {
        this.setState("connecting")

        try {
        // Initialize memory layer asynchronously (tiktoken WASM import)
        await this.ensureMemoryReady()

        await this.player.init()

        this.tts.on({
            onAudio: (samples) => {
                // Drop stale audio from cancelled pipelines
                if (
                    !this.abortController ||
                    this.abortController.signal.aborted
                )
                    return
                // Stop ASR once on first audio — user didn't resume
                // speaking in time, so commit to playback.
                if (
                    !this.asrStoppedForPlayback &&
                    (
                        this.state === "thinking" ||
                        this.state === "speaking"
                    )
                ) {
                    this.asrStoppedForPlayback = true
                    this.callbacks.onLog?.(
                        "TTS first audio — stopping ASR"
                    )
                    this.stopASRSession()
                }
                this.player.enqueue(samples)
            },
            onDone: () => {
                this.callbacks.onLog?.("TTS done for context")
            },
            onError: (err) => this.callbacks.onError?.(err),
            onLog: (msg) => this.callbacks.onLog?.(msg),
        })

        // Best-effort TTS connect — if Cartesia is temporarily down, the
        // service still initializes successfully and ensureConnected() will
        // retry transparently when speech is actually needed.
        try {
            await this.tts.connect()
        } catch (err) {
            this.callbacks.onLog?.(
                `TTS initial connect failed (will retry on first use): ${toError(err).message}`
            )
        }

        // Wire up ASR callbacks
        this.asr.on({
            onInterim: (text) => {
                if (!this.shouldCaptureAudioForASR()) return
                this.callbacks.onUserTranscript?.(text)
                // User resumed speaking while LLM is generating but before TTS
                // audio arrived — cancel premature pipeline, let user finish
                if (
                    (this.state === "thinking" || this.state === "speaking") &&
                    !this.ttsAudioStarted
                ) {
                    this.cancelPrematurePipeline()
                }
            },
            onFinal: (text) => {
                // Ignore in-flight transcripts that arrive after muting —
                // DashScope may finalize buffered speech even after we stop
                // sending audio.
                if (!this.shouldCaptureAudioForASR()) {
                    this.callbacks.onLog?.("Ignoring transcript while muted")
                    return
                }
                this.callbacks.onUserTranscript?.(text)
                this.handleFinalTranscript(text)
            },
            onFinished: () => {
                this.callbacks.onLog?.("ASR session finished")
                // DashScope may end the task on its own (e.g. server-side
                // silence/inactivity timeout). If we're still in listening
                // state, restart ASR so the user isn't stuck with a dead mic.
                if (this.shouldRestartASRForCurrentState()) {
                    this.callbacks.onLog?.("ASR task ended during listening — restarting…")
                    this.startASRWithRetry().catch((retryErr) => {
                        this.callbacks.onLog?.(
                            `ASR restart after task-finished failed: ${toError(retryErr).message}`
                        )
                        if (this.state === "listening" || this.state === "connecting") {
                            this.setState("idle")
                        }
                    })
                }
            },
            onError: (err) => {
                // Suppress DashScope "request timeout" when mic is muted —
                // benign condition (no audio → server timeout) already
                // handled by stopping ASR on mute.
                if (!this.shouldCaptureAudioForASR() && /request timeout/i.test(err.message)) {
                    this.callbacks.onLog?.(
                        `Suppressed ASR timeout (mic muted): ${err.message}`
                    )
                    return
                }
                this.callbacks.onError?.(err)

                // If ASR failed while we were listening (e.g. DashScope
                // task-failed, network drop), the session is dead but the
                // state is still "listening". Restart ASR so the user
                // isn't stuck with a dead mic.
                if (this.shouldRestartASRForCurrentState()) {
                    this.callbacks.onLog?.("ASR error while listening — restarting…")
                    this.startASRWithRetry().catch((retryErr) => {
                        this.callbacks.onLog?.(
                            `ASR restart after error failed: ${toError(retryErr).message}`
                        )
                        if (this.state === "listening" || this.state === "connecting") {
                            this.setState("idle")
                        }
                    })
                }
            },
            onLog: (msg) => this.callbacks.onLog?.(msg),
        })

        // Set a system prompt for the assistant voice-agent persona
        this.conversationHistory = [
            {
                role: "system",
                content: this.buildSystemPrompt(),
            },
        ]

        // Load persisted memory (if storage adapter was provided)
        await this.loadMemoryOnce({
            logMessage: "Voice memory loaded from storage",
        })

        // Start audio level polling
        this.startAudioLevelPolling()

        // Start polling for background task completions
        if (this.backgroundTaskPoller) {
            this.startTaskPolling()
        }

        // Mark text-ready so sendTextOnly() skips its own init
        this.textReadyPromise = Promise.resolve()

        // Speak a greeting so the user knows the agent is ready.
        // continuous mode — no need to startListening() here first,
        // which would just get immediately stopped by the greeting
        // and cause FunASR NO_VALID_AUDIO_ERROR.
        this.continuousMode = true
        this.speakGreeting().catch((e) => {
            this.callbacks.onLog?.("Greeting failed: " + toError(e).message)
            // Ensure ASR starts even if greeting fails — otherwise the
            // agent sits in "idle" with no mic and the user is stuck.
            this.startListening(true).catch(() => {})
        })
        } catch (err) {
            this.setState("idle")
            throw err
        }
    }

    /**
     * Start listening for speech via ASR.
     * @param continuous  If true, listening auto-restarts after each AI response.
     */
    async startListening(continuous = true): Promise<void> {
        if (this.state === "listening" || this.state === "connecting") return

        // If currently speaking, stop playback and ASR
        if (this.state === "speaking" || this.state === "thinking") {
            this.continuationPending = false
            this.ttsAudioStarted = false
            this.asrStoppedForPlayback = false
            ++this.pipelineGeneration
            this.stopASRSession()
            this.abortController?.abort()
            this.abortController = null
            if (this.currentContextId) {
                this.tts.cancel(this.currentContextId).catch(() => {})
                this.currentContextId = null
            }
            this.player.stop()
            await this.player.init()
        }

        this.continuousMode = continuous
        this.setState("connecting")

        try {
            this.callbacks.onLog?.("Starting ASR…")
            await this.startASRWithRetry()
        } catch (err) {
            this.callbacks.onLog?.(`ASR start failed: ${toError(err).message}`)
            this.callbacks.onError?.(toError(err))
            this.setState("idle")
        }
    }

    /** Stop listening and trigger LLM processing with final transcript. */
    async stopListening(): Promise<void> {
        if (this.state !== "listening") return
        try {
            await this.stopASRSessionAwaited()
        } catch (err) {
            this.callbacks.onError?.(toError(err))
        }
    }

    /**
     * Start ASR with retry + exponential backoff.
     * Recovers from transient DashScope connection failures (timeouts,
     * network blips) that would otherwise leave the agent stuck in idle.
     */
    private async startASRWithRetry(maxRetries = 3): Promise<void> {
        const asrOpts = {
            maxSilenceMs:
                this.asrOverrides.maxSilenceMs ?? ASR_MAX_SENTENCE_SILENCE_MS,
            speechNoiseThreshold: this.asrOverrides.speechNoiseThreshold ?? ASR_SPEECH_NOISE_THRESHOLD,
            languageHints: this.asrOverrides.languageHints ?? ASR_LANGUAGE_HINTS,
        }

        let lastError: Error | null = null
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                if (attempt > 1) {
                    const delay = 1000 * attempt
                    this.callbacks.onLog?.(`ASR retry ${attempt}/${maxRetries} in ${delay}ms…`)
                    await new Promise(r => setTimeout(r, delay))
                    // State may have changed during the delay (user interrupted)
                    if (!this.canRetryASRForCurrentState()) return
                }
                await this.asr.start(this.asrApiKey, asrOpts)
                if (this.state === "connecting" && this.shouldCaptureAudioForASR()) {
                    this.setState("listening")
                }
                return // success
            } catch (err) {
                lastError = toError(err)
                this.callbacks.onLog?.(
                    `ASR attempt ${attempt}/${maxRetries} failed: ${lastError.message}`
                )
                if (attempt < maxRetries) {
                    // Force close the stale WebSocket so ensureConnection
                    // creates a fresh one on the next attempt
                    this.destroyASRSession()
                }
            }
        }
        throw lastError ?? new Error("ASR start failed after retries")
    }

    /** Send text directly (for typed input — routes through TTS). */
    async sendText(text: string): Promise<void> {
        const wasContinuous = this.continuousMode
        this.interrupt()
        this.continuousMode = wasContinuous
        this.callbacks.onUserTranscript?.(text)
        await this.processUserInput(text)
    }

    /**
     * Lightweight init for text-only chat (no TTS/ASR/greeting).
     *
     * Sets up system prompt, loads memory from disk, and starts background
     * task polling — everything needed for `sendTextOnly` to work without
     * requiring a full `init()` call.
     *
     * Idempotent — safe to call multiple times, and skipped if `init()`
     * has already run.
     */
    private ensureTextReady(): Promise<void> {
        if (this.textReadyPromise) return this.textReadyPromise

        const p = (async () => {
            // Set up the system prompt if not already done
            if (this.conversationHistory.length === 0) {
                this.conversationHistory = [
                    {
                        role: "system",
                        content: this.buildSystemPrompt(),
                    },
                ]
            }

            await this.loadMemoryOnce({
                logMessage: "Voice memory loaded from storage (text-only)",
            })

            // Start polling for background task completions
            if (this.backgroundTaskPoller) {
                this.startTaskPolling()
            }
        })()

        // If init fails, clear the promise so the next call retries
        // instead of permanently returning a rejected promise.
        this.textReadyPromise = p.catch((err) => {
            this.textReadyPromise = null
            throw err
        })

        return this.textReadyPromise
    }

    /**
     * Send text from the chat drawer — routes through the full voice
     * pipeline (LLM → TTS → playback → restart mic) so voice chat
     * stays active. Also fires `onUserMessage` for chat UI display.
     *
     * Falls back to `ensureTextReady()` when `init()` hasn't been
     * called yet (e.g. text-only first use).
     */
    async sendTextOnly(text: string): Promise<void> {
        await this.ensureTextReady()
        this.callbacks.onUserMessage?.(text)
        await this.sendText(text)
    }

    /** Stop everything and reset to idle. */
    interrupt(): void {
        this.continuousMode = false
        this.continuationPending = false
        this.ttsAudioStarted = false
        this.asrStoppedForPlayback = false
        ++this.pipelineGeneration
        this.stopASRSession()
        this.abortController?.abort()
        this.abortController = null
        if (this.currentContextId) {
            this.tts.cancel(this.currentContextId).catch(() => {})
            this.currentContextId = null
        }
        this.player.stop()
        this.player.init().catch(() => {})
        this.setState("idle")
    }

    /** Clean shutdown. */
    destroy(): void {
        this.stopAudioLevelPolling()
        this.stopTaskPolling()
        this.interrupt()
        this.destroyASRSession()
        this.tts.disconnect()
        this.destroyRuntime()
    }

    // -----------------------------------------------------------------------
    // Internal pipeline
    // -----------------------------------------------------------------------

    private setState(s: AgentState): void {
        this.setAgentState(s)
    }

    /**
     * Recover from an error by returning to listening (continuous mode)
     * or idle. Prevents the agent from getting stuck in "idle" with no
     * mic when a transient LLM error occurs during continuous conversation.
     */
    private restoreListeningOrIdle(): void {
        if (this.continuousMode && !this.isMicMuted) {
            this.callbacks.onLog?.("Error recovery — restarting mic (continuous mode)")
            this.setState("idle")
            this.startListening(true).catch((e) => {
                this.callbacks.onLog?.(`Error recovery ASR restart failed: ${toError(e).message}`)
                this.setState("idle")
            })
        } else {
            this.setState("idle")
        }
    }

    /**
     * Generate and speak a greeting via the LLM → TTS pipeline.
     *
     * Instead of picking from a hardcoded bag, we inject a tool-role
     * instruction and run the full `processUserInput` pipeline.  The
     * LLM produces a natural, varied greeting every time, and the
     * pipeline handles TTS, playback, history tracking, and mic restart
     * automatically.
     */
    private async speakGreeting(): Promise<void> {
        const hasHistory = this.memory
            ? this.memory.getCurrentTurn() > 0
            : this.conversationHistory.length > 1

        if (hasHistory) {
            this.callbacks.onLog?.("Generating returning-user LLM greeting…")
            await this.appendHistoryMessage({
                role: "tool",
                content: JSON.stringify({
                    note: "The user just reconnected to the voice agent. They are a returning user — you have memory of prior sessions (see Conversation Memory above). Generate a brief, natural greeting that acknowledges this is a continuing relationship. Do NOT say this is your first conversation.",
                }),
                reason: "Voice greeting memory sync failed",
                fallbackHistory: this.conversationHistory,
            })

            await this.processUserInput("", true)
            return
        }

        if (this.greetingMessage) {
            await this.speakConfiguredGreetingMessage(this.greetingMessage)
            return
        }

        this.callbacks.onLog?.("Generating LLM greeting…")

        // Inject a greeting instruction (same pattern as
        // drainAndAnnounceCompletions — tool message + processUserInput).
        const instruction = JSON.stringify({
            note: "The user just connected to the voice agent for the first time. Generate a brief, natural greeting (one short sentence). Vary your phrasing — don't repeat the same greeting every time.",
        })
        await this.appendHistoryMessage({
            role: "tool",
            content: instruction,
            reason: "Voice greeting memory sync failed",
            fallbackHistory: this.conversationHistory,
        })

        await this.processUserInput("", true)
    }

    private async speakConfiguredGreetingMessage(message: string): Promise<void> {
        this.callbacks.onLog?.("Speaking configured greeting…")

        await this.appendHistoryMessage({
            role: "assistant",
            content: message,
            reason: "Voice greeting memory sync failed",
            fallbackHistory: this.conversationHistory,
        })

        this.callbacks.onPartialResponse?.(message)
        this.callbacks.onFullResponse?.(message)

        this.abortController = new AbortController()
        const { signal } = this.abortController
        const contextId = `ctx-${++this.contextCounter}-${Date.now()}`
        this.currentContextId = contextId
        this.prepareVoiceTurn()

        await this.tts.sendChunk(contextId, message, true)
        await this.tts.flush(contextId)
        await this.tts.waitForContextDone(contextId)
        await this.player.waitForEnd()

        if (signal.aborted) {
            return
        }

        this.currentContextId = null

        if (this.pendingAnnouncements.length > 0) {
            this.drainAndAnnounceCompletions()
            return
        }

        if (this.continuousMode && !this.isMicMuted) {
            this.callbacks.onLog?.("Playback done — restarting mic")
            this.setState("idle")
            this.startListening(true).catch((e) =>
                this.callbacks.onError?.(toError(e)),
            )
        } else {
            this.setState("idle")
        }
    }

    /**
     * Build the system prompt and append the shared tool surface when tools
     * or bash are available. Shared run-line and bash guidance live in the
     * prompt helper/tool-call prompt modules; this method just composes them.
     */
    private buildSystemPrompt(): string {
        const tools = this.getRegisteredToolDefinitions()
        return buildToolAwareSystemPrompt(this.systemPrompt, tools, {
            allowedRunTargets: this.toolRegistry.getAvailableRunTargets?.() ?? [],
        })
    }

    private handleFinalTranscript(text: string): void {
        // Guard: ignore empty / whitespace-only transcripts from ASR
        if (!text || text.trim().length === 0) return

        if (this.continuationPending) {
            this.mergeIntoLastUserMessage(text)
            this.continuationPending = false
            const mergedText = this.getLastUserContent()
            this.callbacks.onLog?.(`Continuation merged: "${mergedText}"`)
            this.callbacks.onUserTranscript?.(mergedText)
            this.processUserInput(mergedText, true).catch((e) =>
                this.callbacks.onError?.(toError(e))
            )
        } else if (this.state === "listening" || this.state === "idle") {
            this.processUserInput(text).catch((e) =>
                this.callbacks.onError?.(toError(e))
            )
        } else {
            // State is "thinking" or "speaking" with ttsAudioStarted=true
            // (cancellation window already closed). Log, don't silently drop.
            this.callbacks.onLog?.(
                `Transcript dropped (state=${this.state}, ttsAudioStarted=${this.ttsAudioStarted}): "${text.slice(0, 80)}"`
            )
        }
    }

    private cancelPrematurePipeline(): void {
        this.callbacks.onLog?.(
            `User resumed speaking — cancelling premature pipeline (state=${this.state}, ttsAudioStarted=${this.ttsAudioStarted})`
        )
        ++this.pipelineGeneration
        this.abortController?.abort()
        this.abortController = null

        if (this.currentContextId) {
            this.tts.cancel(this.currentContextId).catch(() => {})
            this.currentContextId = null
        }

        this.player.stop()
        this.player.init().catch(() => {})

        // Remove the incomplete assistant message from whichever store is active
        if (this.memory) {
            if (this.memory.getLastMessageRole() === "assistant") {
                this.memory.popLastMessage()
            }
        } else if (
            this.conversationHistory.length > 0 &&
            this.conversationHistory[this.conversationHistory.length - 1]
                .role === "assistant"
        ) {
            this.conversationHistory.pop()
        }

        this.ttsAudioStarted = false
        this.asrStoppedForPlayback = false
        this.continuationPending = true
        this.setState("listening")
    }

    private mergeIntoLastUserMessage(text: string): void {
        if (this.memory) {
            const last = this.memory.getLastMessageByRole("user")
            if (last) {
                this.memory.updateLastMessageByRole("user", last.content + " " + text)
            } else {
                this.memory.appendMessage("user", text)
            }
            return
        }

        for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
            if (this.conversationHistory[i].role === "user") {
                this.conversationHistory[i] = {
                    ...this.conversationHistory[i],
                    content: this.conversationHistory[i].content + " " + text,
                }
                return
            }
        }
        this.conversationHistory.push({ role: "user", content: text })
    }

    private getLastUserContent(): string {
        if (this.memory) {
            return this.memory.getLastMessageByRole("user")?.content ?? ""
        }
        for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
            if (this.conversationHistory[i].role === "user") {
                return this.conversationHistory[i].content
            }
        }
        return ""
    }

    /**
     * Core pipeline: run a shared streamed LLM/tool turn, then settle
     * playback and microphone lifecycle for the voice transport.
     */
    private async processUserInput(
        text: string,
        skipHistoryPush = false,
        skipTTS = false
    ): Promise<void> {
        this.callbacks.onLog?.(`Processing user input: "${text}"`)
        this.abortController = new AbortController()
        const { signal } = this.abortController

        await this.turnController.run({
            text,
            skipHistoryPush,
            skipSpeech: skipTTS,
            signal,
        })
    }

    private prepareVoiceTurn(): void {
        this.setState("thinking")
        this.ttsAudioStarted = false
        this.asrStoppedForPlayback = false
    }

    private createVoiceTurnOptions(
        text: string,
        skipHistoryPush: boolean,
        skipSpeech: boolean,
        signal: AbortSignal,
    ) {
        const contextId = `ctx-${++this.contextCounter}-${Date.now()}`
        this.currentContextId = contextId

        return {
            text,
            contextId,
            signal,
            skipHistoryPush,
            skipSpeech,
        }
    }

    // -----------------------------------------------------------------------
    // Background task polling + announcement queue
    // -----------------------------------------------------------------------

    /** Start periodic polling for completed background tasks. */
    private startTaskPolling(): void {
        if (this.taskPollingTimer) return
        this.taskPollingTimer = setInterval(
            () => this.pollBackgroundTasks(),
            VoiceAgentService.TASK_POLL_INTERVAL_MS
        )
    }

    /** Stop polling for background tasks. */
    private stopTaskPolling(): void {
        if (this.taskPollingTimer) {
            clearInterval(this.taskPollingTimer)
            this.taskPollingTimer = null
        }
    }

    /**
     * Poll for completed background tasks (via the injected poller).
     * Queues results for announcement and triggers immediate announcement
     * if the agent is currently idle.
     */
    private async pollBackgroundTasks(): Promise<void> {
        if (!this.backgroundTaskPoller) return
        try {
            const completed = await this.backgroundTaskPoller()
            if (completed.length === 0) return

            this.pendingAnnouncements.push(...completed)
            this.callbacks.onLog?.(
                `${completed.length} background task(s) completed, queued for announcement`
            )

            // Notify UI via callbacks
            for (const task of completed) {
                this.callbacks.onBackgroundTaskComplete?.(task)
            }

            // If idle or listening, announce immediately. In
            // continuous mode the agent goes straight to "listening"
            // after speaking, so checking only "idle" would miss
            // tasks that complete while the mic is active.
            if (this.state === "idle" || this.state === "listening") {
                this.drainAndAnnounceCompletions()
            }
            // Otherwise, announcements drain after the current turn finishes
        } catch (err) {
            this.callbacks.onLog?.(
                `Background task poll error: ${toError(err).message}`
            )
        }
    }

    /**
     * Drain all pending background task completions into conversation
     * history and trigger the LLM to generate a spoken announcement.
     *
     * Called from two places:
     *   1. In `pollBackgroundTasks` when agent is idle
     *   2. In the post-playback hook, before restarting mic
     *
     * If more tasks complete while the announcement is playing, they'll
     * be caught by the next drain cycle (post-playback hook).
     */
    private drainAndAnnounceCompletions(): void {
        if (this.pendingAnnouncements.length === 0) return

        // If ASR is active (listening state), stop it before entering
        // the announcement pipeline. processUserInput's post-playback
        // hook restarts the mic in continuous mode after the announcement
        // plays. We don't await — the stop begins asynchronously, and
        // by the time any onFinal callback fires, processUserInput
        // below has already set state to "thinking" synchronously,
        // so handleFinalTranscript won't process it.
        if (this.state === "listening") {
            this.stopASRSession()
        }

        const announcements = this.pendingAnnouncements.splice(0)

        for (const ann of announcements) {
            const toolContent = JSON.stringify({
                tool: ann.tool,
                taskId: ann.taskId,
                task: ann.task,
                success: ann.success,
                output: ann.output,
                ...(ann.error && { error: ann.error }),
                note: "This background task has completed. Briefly announce the result to the user.",
            })
            void this.appendHistoryMessage({
                role: "tool",
                content: toolContent,
                reason: "Voice announcement save failed",
                fallbackHistory: this.conversationHistory,
            })
        }

        this.callbacks.onLog?.(
            `Announcing ${announcements.length} completed task(s)…`
        )

        // Re-call LLM — it sees the tool results and produces a spoken summary.
        this.processUserInput("", true)
            .catch((e) => this.callbacks.onError?.(toError(e)))
            .finally(() => {
                // restoreListeningOrIdle() handles most error paths now,
                // but this .finally() is a safety net in case processUserInput
                // resolves successfully but drains exit before startListening.
                // startListening() guards on state==="listening" so double
                // calls are no-ops.
                if (this.state === "idle" && this.continuousMode && !this.isMicMuted) {
                    this.startListening(true).catch(() => {})
                }
            })
    }

}

export interface SopranoApiKeys {
    qwenApiKey: string
    cartesiaApiKey: string
    asrApiKey?: string
}

export interface CreateSopranoVoiceAgentOptions extends VoiceAgentDeps {}

export function createSopranoVoiceAgent(
    options: CreateSopranoVoiceAgentOptions,
): VoiceAgentService {
    return new VoiceAgentService(options)
}
