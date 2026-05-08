/**
 * Shared type definitions for the voice agent pipeline.
 *
 * These interfaces allow platform-specific implementations of audio
 * playback and speech recognition while sharing the orchestration logic.
 */

import type { ToolCall } from "./types"

// ---------------------------------------------------------------------------
// Agent state machine
// ---------------------------------------------------------------------------

export type AgentState =
    | "idle"
    | "connecting"
    | "listening"
    | "thinking"
    | "speaking"

export const AGENT_STATES: readonly AgentState[] = [
    "idle",
    "connecting",
    "listening",
    "thinking",
    "speaking",
] as const

// ---------------------------------------------------------------------------
// Background tool results
// ---------------------------------------------------------------------------

/** Result from a background tool task (e.g. Hammer) that completed async. */
export interface BackgroundToolResult {
    taskId: string
    /** Tool name (e.g. "HammerAgent"). */
    tool: string
    /** The original task description. */
    task: string
    success: boolean
    output: string
    error?: string
}

export interface ConversationMessage {
    role: "user" | "assistant"
    text: string
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

export interface VoiceAgentCallbacks {
    onStateChange?: (state: AgentState) => void
    /** Fired as the LLM streams tokens. */
    onPartialResponse?: (text: string) => void
    /** Fired once the full LLM response is assembled. */
    onFullResponse?: (text: string) => void
    /** The user's final recognized speech text. */
    onUserTranscript?: (text: string) => void
    /** Fired when a finalized user message should be recorded (e.g. typed input). */
    onUserMessage?: (text: string) => void
    /** Fired when the LLM emits the final selected tool call for the turn. */
    onToolCall?: (selectedToolCall: ToolCall) => void
    /** Fired if structured command text was detected in the stream but could not be parsed. */
    onToolCallError?: (error: Error, rawCommandText: string) => void
    /** Fired when a background task completes (before LLM announcement). */
    onBackgroundTaskComplete?: (result: BackgroundToolResult) => void
    /** General log messages. */
    onLog?: (msg: string) => void
    /** Error callback. */
    onError?: (err: Error) => void
    /** Fired every animation frame with the current audio level (0–1). */
    onAudioLevel?: (level: number) => void
}

export interface ASRCallbacks {
    /** Interim (partial) transcript update. */
    onInterim?: (text: string) => void
    /** Final (sentence-end) transcript. */
    onFinal?: (text: string) => void
    /** ASR session finished. */
    onFinished?: () => void
    /** Error. */
    onError?: (err: Error) => void
    /** Log messages. */
    onLog?: (msg: string) => void
}

export interface CartesiaTTSCallbacks {
    /** Called with decoded Float32 PCM audio samples ready to play. */
    onAudio?: (samples: Float32Array) => void
    /** Called when all audio for a context has been received. */
    onDone?: (contextId: string) => void
    /** Called on errors. */
    onError?: (error: Error) => void
    /** Informational log messages. */
    onLog?: (msg: string) => void
}

// ---------------------------------------------------------------------------
// Platform abstractions
// ---------------------------------------------------------------------------

/** Platform-agnostic audio playback interface. */
export interface IAudioPlayer {
    /** Initialize audio subsystem. Must be called from a user gesture on web. */
    init(): Promise<void>
    /** Enqueue Float32 PCM samples for playback. */
    enqueue(samples: Float32Array): void
    /** Interrupt and stop all playback. */
    stop(): void
    /** Get the current output audio level (0–1). */
    getAudioLevel(): number
    /** True when audio has been queued and hasn't finished yet. */
    readonly isPlaying: boolean
    /** Returns a promise that resolves when all queued audio has finished. */
    waitForEnd(): Promise<void>
}

/** Platform-agnostic ASR (speech recognition) interface. */
export interface IASRService {
    /** Register event callbacks. */
    on(callbacks: ASRCallbacks): void
    /**
     * Start real-time ASR.
     * @param apiKey  DashScope API key
     * @param opts    ASR configuration options
     */
    start(apiKey: string, opts?: ASRStartOptions): Promise<void>
    /** Stop the current ASR session. */
    stop(): Promise<void>
    /** Hard cleanup — release all resources. */
    destroy(): void
    /** Mute/unmute the microphone. When muted, no audio is sent to the ASR engine. */
    setMicMuted?(muted: boolean): void
    /** Whether the microphone is currently muted. */
    readonly isMicMuted?: boolean
}

export interface ASRStartOptions {
    maxSilenceMs?: number
    speechNoiseThreshold?: number | null
    languageHints?: readonly string[]
}

// ---------------------------------------------------------------------------
// UI helpers — shared state-to-display mapping
// ---------------------------------------------------------------------------

/** Human-readable label for each agent state (e.g. "Listening…"). */
export function getAgentStateLabel(state: AgentState): string {
    switch (state) {
        case "connecting":
            return "Connecting…"
        case "listening":
            return "Listening…"
        case "thinking":
            return "Thinking…"
        case "speaking":
            return "Speaking…"
        default:
            return "Idle"
    }
}

/** Semantic color (hex) for each agent state. */
export function getAgentStateColor(state: AgentState): string {
    switch (state) {
        case "connecting":
            return "#f39c12"
        case "listening":
            return "#3498db"
        case "thinking":
            return "#9b59b6"
        case "speaking":
            return "#2ecc71"
        default:
            return "#95a5a6"
    }
}
