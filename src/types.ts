/**
 * Core type definitions shared across mjolno apps.
 *
 * These types are intentionally decoupled from any CLI or desktop-specific
 * concerns so they can be consumed by both apps/hammer and apps/tauri.
 */

export interface ChatMessage {
    role: "system" | "user" | "assistant" | "tool"
    content: string
}

export interface FetchResponseLike {
    ok: boolean
    status: number
    text(): Promise<string>
    json(): Promise<unknown>
    body?: unknown
}

export type FetchLike = (url: string, init?: any) => Promise<FetchResponseLike>

/** Configuration for an LLM provider endpoint. */
export interface LLMProviderConfig {
    apiKey: string
    baseUrl: string
    model: string
    /** Extra headers merged into every request (e.g. HTTP-Referer for OpenRouter). */
    extraHeaders?: Record<string, string>
    /** Provider-specific request fields merged into the chat completion payload. */
    extraBody?: Record<string, unknown>
    /** Custom fetch implementation (e.g. expo/fetch for RN streaming support). */
    fetchImpl?: FetchLike
}

/** Options for a single chat completion request. */
export interface LLMRequestOptions {
    messages: ChatMessage[]
    temperature?: number
    maxTokens?: number
    stream?: boolean
    frequencyPenalty?: number
    presencePenalty?: number
    extraBody?: Record<string, unknown>
    signal?: AbortSignal
}

/** Callbacks invoked during streaming. All optional – pass only what you need. */
export interface StreamCallbacks {
    /**
     * Fired for every content token received.
     *
     * Return `true` to signal early termination — the stream will be
     * aborted cleanly and `onComplete` will still fire with the content
     * accumulated so far.
     */
    onToken?: (token: string) => void | boolean
    /**
     * Fired for every reasoning/thinking token received
     * (delta.reasoning_content). Called before any content tokens arrive
     * for models that emit a thinking phase (e.g. Qwen 3+).
     */
    onReasoningToken?: (token: string) => void
    /**
     * Fired once when the first SSE data chunk arrives from the model.
     * Useful for closing premature-cancellation windows: Qwen 3+ models
     * send reasoning tokens (delta.reasoning_content) for several seconds
     * before any delta.content arrives, so onToken alone is insufficient
     * to detect that the model is actively processing.
     */
    onStreamStart?: () => void
    /** Fired once when the stream finishes (includes full concatenated content). */
    onComplete?: (fullContent: string, finishReason: string) => void
    /** Fired on recoverable stream errors (e.g. frozen stream detected). */
    onError?: (error: Error) => void
    /** Informational log messages (replaces console.log / chalk in shared code). */
    onLog?: (message: string, level: "info" | "warn" | "error") => void
}

/** Result returned by LLMClient.chat(). */
export interface LLMClientResponse {
    content: string
    finishReason: string
    usage?: {
        promptTokens: number
        completionTokens: number
    }
}

export type ToolDataPrimitive = string | number | boolean | null

export type ToolDataValue =
    | ToolDataPrimitive
    | ToolDataValue[]
    | { [key: string]: ToolDataValue }

export interface ToolDataSchema {
    type?: string | string[]
    description?: string
    enum?: readonly ToolDataPrimitive[]
    items?: ToolDataSchema
    properties?: Record<string, ToolDataSchema>
    required?: string[]
    additionalProperties?: boolean | ToolDataSchema
    default?: ToolDataValue
}

export interface ToolParameterDefinition {
    type: string | string[]
    description: string
    required?: boolean
    positional?: boolean
    enum?: readonly ToolDataPrimitive[]
    items?: ToolDataSchema
    properties?: Record<string, ToolDataSchema>
    additionalProperties?: boolean | ToolDataSchema
    default?: ToolDataValue
}

export type ToolMemoryNoteScope = "none" | "tool"

export type ToolMemoryNoteKind = "workflow" | "operation" | "change"

export type ToolMemoryEvidenceKind = "codebase" | "research"

export type ToolMemoryCitationKind =
    | "path"
    | "query"
    | "url"
    | "command"

export interface ToolMemoryNotePolicy {
    scope: ToolMemoryNoteScope
    kind?: ToolMemoryNoteKind
}

export interface ToolMemoryEvidencePolicy {
    kind: ToolMemoryEvidenceKind
    citation?: ToolMemoryCitationKind
}

export interface ToolMemoryMetadata {
    note?: ToolMemoryNotePolicy
    evidence?: ToolMemoryEvidencePolicy[]
}

export interface ToolDefinitionMetadata {
    category: string
    capabilities: string[]
    requirements?: string[]
    memory?: ToolMemoryMetadata
    source?: "builtin" | "mcp"
    mcpServerName?: string
}

export interface ToolDefinition {
    name: string
    description: string
    usageExample?: string
    parameters: Record<string, ToolParameterDefinition>
    metadata?: ToolDefinitionMetadata
}

export interface ToolCall {
    name: string
    parameters: Record<string, any>
    kind?: "tool" | "bash" | "background_bash"
    rawInvocation?: string
    truncated?: boolean
}

export interface ToolResult {
    success: boolean
    error?: string
    [key: string]: any
}

export type LoopOutcome = "continue" | "success" | "failure"

export interface LLMRequest {
    role: "agent"
    task: string
    tools?: ToolDefinition[]
    action_count: number
}

export interface LLMResponse {
    content: string
    reasoning: string
    scratchpad?: string
    selectedToolCall?: ToolCall
    outcome: LoopOutcome
    finishReason?: string
}

export type ProviderName =
    | "qwen-max"
    | "qwen-plus"
    | "openrouter-claude"
    | "openrouter-gemini"
    | "minimax"
    | "minimax-her"
    | "chatglm"
    | "kimi"
    | "doubao"
