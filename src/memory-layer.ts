/**
 * Base Memory Layer — shared infrastructure for memory compaction.
 *
 * Both workspace agents and the Voice Agent use episodic compaction with
 * structured state extraction. This module provides the shared skeleton:
 *
 * - Generic base class (`BaseMemoryLayer<TState, TMessage>`)
 * - Shared types: `MemoryMessage`, `CompactionCursor`, `MemoryStorage`, etc.
 * - Token estimation strategy (pluggable: tiktoken vs char-based)
 * - Shared algorithms: sliding-window, buildMessages, triggerCompaction
 *
 * Core Invariants (enforced by this base):
 * 1. Raw history is NEVER modified in-place; entries are pruned after compaction
 * 2. Compaction is a state transition, not a view transform
 * 3. Compressed state is NEVER recursively summarized
 * 4. State compaction removes obsolete structure but preserves meaning
 * 5. buildMessages is a PURE function (cache writes are benign memoization)
 */

import type { ChatMessage } from "./types"

// ============================================================================
// Compaction LLM Interface
// ============================================================================

/**
 * Minimal interface for an LLM client used during compaction.
 * Kept abstract so memory-layer has no hard dependency on LLMClient.
 * The shared `LLMClient` class satisfies this interface.
 */
export interface CompactionLLMClient {
    chat(
        options: {
            messages: ChatMessage[]
            temperature?: number
            maxTokens?: number
            stream?: boolean
        },
    ): Promise<{ content: string }>
}

// ============================================================================
// Shared Types
// ============================================================================

/** Base message stored in raw history. */
export interface MemoryMessage {
    id: string
    role: "system" | "user" | "assistant" | "tool"
    content: string
    timestamp: number
    /** Monotonic message counter. */
    turn: number
    /** Cached token count for this message (computed once at creation). */
    tokenCount: number
}

export interface MemoryProvenance {
    source: "rule" | "llm" | "tool" | "assistant" | "user"
    detail?: string
}

/** Tracks how far compaction has progressed. */
export interface CompactionCursor {
    lastCompactedTurn: number
}

// ============================================================================
// Token Estimation
// ============================================================================

/**
 * Strategy for estimating token counts.
 *
 * - Workspace agents share the same estimator strategy by default
 * - CharTokenEstimator is the cross-platform baseline implementation
 */
export interface TokenEstimator {
    /** Estimate token count for a string. */
    estimateTokens(text: string): number
    /** Dispose any resources (e.g. tiktoken WASM). No-op by default. */
    dispose?(): void
}

/**
 * Character-based token estimator.
 * GPT/Claude average ~4 chars/token for English; 3.5 is conservative.
 */
export class CharTokenEstimator implements TokenEstimator {
    constructor(private charsPerToken = 3.5) {}

    estimateTokens(text: string): number {
        return Math.ceil(text.length / this.charsPerToken)
    }
}

// ============================================================================
// Persistence
// ============================================================================

/**
 * Platform-agnostic persistence adapter (generic over data shape).
 *
 * Implementations: LocalStorageAdapter, HttpAdapter, ExpoFileAdapter,
 * and the Bun-server FileMemoryAdapter.
 */
export interface MemoryStorage<T = unknown> {
    load(): Promise<T | null>
    save(data: T): Promise<void>
    clear(): Promise<void>
}

/** Base shape of persisted memory data. Apps may extend with extra fields. */
export interface PersistedMemoryData<
    TState = unknown,
    TMessage extends MemoryMessage = MemoryMessage,
> {
    rawHistory: TMessage[]
    compressedState: TState
    compactionCursor: CompactionCursor
    currentTurn: number
    compactionCount: number
}

// ============================================================================
// Configuration
// ============================================================================

export interface MemoryLayerConfig {
    /** Trigger compaction when estimated context exceeds this many tokens. */
    compactionTokenThreshold: number
    /** Keep at least this many tokens of recent messages uncompressed. */
    protectedContextTokens: number
    /** Maximum tokens for the rendered compressed state. */
    stateBudgetTokens: number
    /** Hard cap on raw history entries (safety net). */
    maxRawHistory: number
    /** Minimum turns between compaction attempts. */
    compactionDebounceTurns: number
    /** Baseline token overhead for system prompt (conservative estimate). */
    systemPromptOverhead: number
    /** Strategy for estimating token counts. */
    tokenEstimator: TokenEstimator
    /** Optional LLM client for enhanced compaction (e.g. MiniMax). */
    compactionClient?: CompactionLLMClient
    /** Temperature for compaction LLM calls (default 0.1). */
    compactionTemperature?: number
    /** Max tokens for compaction LLM output (default 4096). */
    compactionMaxTokens?: number
}

// ============================================================================
// Base Memory Layer
// ============================================================================

/**
 * Abstract base class for memory layers with episodic compaction.
 *
 * Subclasses provide:
 * - Message creation (TMessage construction with app-specific fields)
 * - Compaction logic (extraction plus pruning obsolete state entries)
 * - State rendering (converting TState to human-readable text)
 *
 * The base class provides:
 * - Append-only raw history with hard cap enforcement
 * - Token-budgeted sliding window for recent messages
 * - buildMessages assembly (system → state → recent)
 * - Compaction triggering (debounce, threshold check, prune)
 * - Rendered state caching with invalidation
 * - Persistence lifecycle (save/load/clear)
 * - Accessors for metrics
 */
export abstract class BaseMemoryLayer<
    TState,
    TMessage extends MemoryMessage,
> {
    protected rawHistory: TMessage[] = []
    protected compressedState: TState
    protected compactionCursor: CompactionCursor = { lastCompactedTurn: 0 }
    protected currentTurn = 0
    protected compactionCount = 0

    /** Rendered state cache (invalidated on mutation). */
    protected renderedStateCache: string | null = null
    protected stateTokenCountCache: number | null = null

    /** Debounce: turns since last compaction attempt. */
    protected lastCompactionAttempt = 0

    /**
     * Static context: rules/instructions injected once, always included in
     * buildMessages, never compressed or pruned.
     */
    protected staticContext: string | null = null

    constructor(
        protected readonly config: MemoryLayerConfig,
        initialState: TState,
    ) {
        this.compressedState = initialState
    }

    // -----------------------------------------------------------------------
    // Abstract Methods (subclass-specific)
    // -----------------------------------------------------------------------

    /**
     * Create a TMessage from the given parts.
     * Subclass should compute and cache the tokenCount (or use tokenEstimator).
     */
    protected abstract createMessage(
        id: string,
        role: "system" | "user" | "assistant" | "tool",
        content: string,
        turn: number,
        timestamp: number,
    ): TMessage

    /**
    * Extract structured facts from raw messages into compressedState and
    * enforce any dedupe, aging, and hard-cap rules for the durable state.
    * This is deterministic pattern matching, NOT LLM summarization.
     * Should mutate `this.compressedState` in place.
     */
    protected abstract performCompaction(messages: TMessage[]): void

    /**
     * Render compressedState as human-readable structured text for LLM context.
     * Result is cached — only called when cache is invalidated.
     */
    protected abstract renderState(): string

    /**
     * Create a fresh initial state (used by clear()).
     */
    protected abstract createInitialState(): TState

    // -----------------------------------------------------------------------
    // Optional Hooks
    // -----------------------------------------------------------------------

    /** Label for the compressed state section in buildMessages. */
    protected getStateLabel(): string {
        return "Memory"
    }

    /** Placeholder user message when no user message precedes the first assistant. */
    protected getResumePlaceholder(): string {
        return "Continue."
    }

    /** Called after a message is appended (e.g. for failure tracking). */
    protected onMessageAppended(_msg: TMessage): void {}

    /** Called after compaction + prune (e.g. for extra persistence). */
    protected onCompactionComplete(): void {}

    /**
     * LLM-enhanced compaction. Override in subclass to provide LLM-based
     * summarization using `this.config.compactionClient`.
     *
     * When this returns `true`, the pattern-matching `performCompaction()`
     * is skipped. When it returns `false` (or throws), pattern matching
     * runs as a deterministic fallback.
     */
    protected async performLLMCompaction(_messages: TMessage[]): Promise<boolean> {
        return false
    }

    // -----------------------------------------------------------------------
    // Message Management
    // -----------------------------------------------------------------------

    /** Append a message to raw history. Returns the message ID. */
    appendMessage(
        role: "system" | "user" | "assistant" | "tool",
        content: string,
    ): string {
        this.currentTurn++
        const id = `msg_${this.currentTurn}_${Date.now()}`
        const timestamp = Date.now()
        const msg = this.createMessage(id, role, content, this.currentTurn, timestamp)
        this.rawHistory.push(msg)

        // Hard cap enforcement
        if (this.rawHistory.length > this.config.maxRawHistory) {
            const excess = this.rawHistory.length - this.config.maxRawHistory
            const lastPrunedTurn = this.rawHistory[excess - 1].turn
            this.rawHistory.splice(0, excess)
            if (this.compactionCursor.lastCompactedTurn < lastPrunedTurn) {
                this.compactionCursor.lastCompactedTurn = lastPrunedTurn
            }
        }

        this.onMessageAppended(msg)
        return id
    }

    // -----------------------------------------------------------------------
    // Build Messages (PURE — no state mutations except benign cache writes)
    // -----------------------------------------------------------------------

    /**
     * Build the message array for LLM context.
     *
     * Assembly order:
     * 1. System prompt
     * 2. Static context (if any)
     * 3. Compressed state
     * 4. Recent raw messages (token-budgeted sliding window)
     *
     * Post-processing:
     * - Tool messages converted to user role ("Tool result: …")
     * - Consecutive same-role messages merged
     * - Guard: user message inserted before first assistant if missing
     * - Leading system messages merged into one
     */
    buildMessages(systemPrompt: string): ChatMessage[] {
        const messages: ChatMessage[] = []

        // 1. System prompt
        messages.push({ role: "system", content: systemPrompt })

        // 2. Static context (rules/instructions — never compressed)
        if (this.staticContext) {
            messages.push({ role: "system", content: this.staticContext })
        }

        // 3. Compressed state
        const stateBlock = this.getCachedRenderedState()
        if (stateBlock) {
            messages.push({
                role: "system",
                content: `# ${this.getStateLabel()}\n\n${stateBlock}`,
            })
        }

        // 4. Recent messages (token-budget sliding window)
        const recent = this.getRecentMessages()
        for (const msg of recent) {
            if (msg.role === "system") continue

            const role = msg.role === "tool" ? "user" : msg.role
            const content =
                msg.role === "tool"
                    ? `Tool result: ${msg.content}`
                    : msg.content

            // Merge consecutive same-role messages (API alternation requirement)
            const last = messages[messages.length - 1]
            if (last && last.role === role) {
                last.content += "\n\n" + content
            } else {
                messages.push({ role, content })
            }
        }

        // Guard: ensure a user message before first assistant
        const firstNonSystem = messages.findIndex((m) => m.role !== "system")
        if (firstNonSystem !== -1 && messages[firstNonSystem].role === "assistant") {
            messages.splice(firstNonSystem, 0, {
                role: "user",
                content: this.getResumePlaceholder(),
            })
        }

        // Merge leading system messages into one
        const systemParts: string[] = []
        let sysCount = 0
        for (const msg of messages) {
            if (msg.role === "system") {
                systemParts.push(msg.content)
                sysCount++
            } else break
        }
        if (sysCount > 1) {
            messages.splice(0, sysCount, {
                role: "system",
                content: systemParts.join("\n\n---\n\n"),
            })
        }

        return messages
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    /** Trigger compaction if conversation context exceeds token threshold. */
    async triggerCompactionIfNeeded(): Promise<void> {
        // Debounce
        if (
            this.currentTurn - this.lastCompactionAttempt <
            this.config.compactionDebounceTurns
        ) {
            return
        }

        // Compute recent messages once (used for both threshold check and boundary)
        const recentMessages = this.getRecentMessages()
        const recentTokens = recentMessages.reduce(
            (sum, m) => sum + m.tokenCount,
            0,
        )
        const stateContent = this.getCachedRenderedState()
        const stateTokens = stateContent
            ? this.config.tokenEstimator.estimateTokens(stateContent)
            : 0
        const staticTokens = this.staticContext
            ? this.config.tokenEstimator.estimateTokens(this.staticContext)
            : 0
        const estimated = recentTokens + stateTokens + staticTokens + this.config.systemPromptOverhead
        if (estimated < this.config.compactionTokenThreshold) return

        // Find protected window boundary
        const protectedTurnBoundary =
            recentMessages.length > 0
                ? recentMessages[0].turn
                : this.currentTurn + 1

        const compactionBoundary = Math.max(
            this.compactionCursor.lastCompactedTurn + 1,
            1,
        )

        const toCompact = this.rawHistory.filter(
            (m) =>
                m.turn >= compactionBoundary &&
                m.turn < protectedTurnBoundary,
        )

        if (toCompact.length === 0) {
            this.lastCompactionAttempt = this.currentTurn
            return
        }

        // Perform compaction: try LLM first, fall back to pattern matching
        let llmSucceeded = false
        if (this.config.compactionClient) {
            try {
                llmSucceeded = await this.performLLMCompaction(toCompact)
            } catch {
                // LLM compaction failed — fall through to pattern matching
            }
        }
        if (!llmSucceeded) {
            this.performCompaction(toCompact)
        }

        // Advance cursor & bookkeeping
        this.compactionCursor.lastCompactedTurn = protectedTurnBoundary - 1
        this.lastCompactionAttempt = this.currentTurn
        this.compactionCount++
        this.invalidateStateCache()

        // Prune compacted messages from rawHistory
        this.rawHistory = this.rawHistory.filter(
            (m) => m.turn >= protectedTurnBoundary,
        )

        // Hook for subclass (e.g. persistence, extra logging)
        this.onCompactionComplete()
    }

    // -----------------------------------------------------------------------
    // Sliding Window
    // -----------------------------------------------------------------------

    /** Get recent messages that fit within the protected token budget. */
    protected getRecentMessages(): TMessage[] {
        const messages: TMessage[] = []
        let tokenCount = 0

        for (let i = this.rawHistory.length - 1; i >= 0; i--) {
            const msg = this.rawHistory[i]
            if (msg.role === "system") continue

            const msgTokens = msg.tokenCount
            if (
                tokenCount + msgTokens > this.config.protectedContextTokens &&
                messages.length > 0
            ) {
                break
            }

            messages.unshift(msg)
            tokenCount += msgTokens
        }

        return messages
    }

    // -----------------------------------------------------------------------
    // Token Estimation
    // -----------------------------------------------------------------------

    /** Estimate total context tokens (recent + state + system overhead). */
    estimateContextTokens(): number {
        const recentTokens = this.getRecentMessages().reduce(
            (sum, m) => sum + m.tokenCount,
            0,
        )
        const stateContent = this.getCachedRenderedState()
        const stateTokens = stateContent
            ? this.config.tokenEstimator.estimateTokens(stateContent)
            : 0
        const staticTokens = this.staticContext
            ? this.config.tokenEstimator.estimateTokens(this.staticContext)
            : 0
        return recentTokens + stateTokens + staticTokens + this.config.systemPromptOverhead
    }

    /** Estimate compressed state tokens. Cached alongside rendered state. */
    estimateStateTokens(): number {
        if (this.stateTokenCountCache !== null) return this.stateTokenCountCache
        const stateContent = this.getCachedRenderedState()
        if (!stateContent) {
            this.stateTokenCountCache = 0
            return 0
        }
        this.stateTokenCountCache = this.config.tokenEstimator.estimateTokens(stateContent)
        return this.stateTokenCountCache
    }

    // -----------------------------------------------------------------------
    // State Rendering (cached)
    // -----------------------------------------------------------------------

    /** Get rendered state, using cache when valid. */
    protected getCachedRenderedState(): string {
        if (this.renderedStateCache !== null) return this.renderedStateCache
        this.renderedStateCache = this.renderState()
        return this.renderedStateCache
    }

    // -----------------------------------------------------------------------
    // Cache Invalidation
    // -----------------------------------------------------------------------

    /** Invalidate rendered state cache (call after any state mutation). */
    invalidateStateCache(): void {
        this.renderedStateCache = null
        this.stateTokenCountCache = null
    }

    // -----------------------------------------------------------------------
    // Static Context
    // -----------------------------------------------------------------------

    /**
     * Set static context (rules/instructions injected once into every
     * buildMessages call). Never compressed or pruned.
     */
    setStaticContext(content: string): void {
        this.staticContext = content
    }

    /** Check if static context has been set. */
    hasStaticContext(): boolean {
        return this.staticContext !== null
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /** Serialize base memory state. Subclasses can extend. */
    serializeBase(): PersistedMemoryData<TState, TMessage> {
        return {
            rawHistory: [...this.rawHistory],
            compressedState: structuredClone(this.compressedState),
            compactionCursor: { ...this.compactionCursor },
            currentTurn: this.currentTurn,
            compactionCount: this.compactionCount,
        }
    }

    /** Restore base memory state from persisted data. */
    loadBase(data: PersistedMemoryData<TState, TMessage>): void {
        if (!Array.isArray(data.rawHistory)) {
            throw new Error("Invalid rawHistory: must be an array")
        }

        this.rawHistory = data.rawHistory
        this.compressedState = data.compressedState
        this.compactionCursor = {
            lastCompactedTurn:
                typeof data.compactionCursor?.lastCompactedTurn === "number"
                    ? data.compactionCursor.lastCompactedTurn
                    : 0,
        }

        const maxTurn = this.rawHistory.reduce(
            (max, m) => Math.max(max, m.turn ?? 0),
            0,
        )
        this.currentTurn = Math.max(data.currentTurn ?? 0, maxTurn)
        this.compactionCount = data.compactionCount ?? 0
        this.invalidateStateCache()
    }

    /**
     * Sanitize raw history messages loaded from persistence.
     *
    * Guards against corrupted or older persisted data by providing safe defaults
     * for all base MemoryMessage fields. Subclasses can extend to handle
     * app-specific fields (e.g. charCount, toolCallId).
     */
    protected sanitizeHistory(
        rawHistory: Array<Partial<TMessage> & Record<string, any>>,
    ): TMessage[] {
        return rawHistory.map((msg, idx) => ({
            ...msg,
            id: msg.id || `recovered_${idx}`,
            role: msg.role || ("user" as const),
            content: msg.content ?? "",
            turn: typeof msg.turn === "number" ? msg.turn : idx,
            timestamp:
                typeof msg.timestamp === "number" ? msg.timestamp : Date.now(),
            tokenCount:
                typeof msg.tokenCount === "number"
                    ? msg.tokenCount
                    : this.config.tokenEstimator.estimateTokens(
                          msg.content ?? "",
                      ),
        })) as TMessage[]
    }

    /** Reset all state. */
    reset(): void {
        this.rawHistory = []
        this.compressedState = this.createInitialState()
        this.compactionCursor = { lastCompactedTurn: 0 }
        this.currentTurn = 0
        this.compactionCount = 0
        this.lastCompactionAttempt = 0
        this.staticContext = null
        this.invalidateStateCache()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    getCurrentTurn(): number {
        return this.currentTurn
    }

    getCompactionCount(): number {
        return this.compactionCount
    }

    getRawHistoryLength(): number {
        return this.rawHistory.length
    }

    getRawHistory(): TMessage[] {
        return [...this.rawHistory]
    }

    getCompressedState(): TState {
        return structuredClone(this.compressedState)
    }

    getCompactionCursor(): CompactionCursor {
        return { ...this.compactionCursor }
    }

    getLastMessageRole(): string | null {
        if (this.rawHistory.length === 0) return null
        return this.rawHistory[this.rawHistory.length - 1].role
    }

    getLastMessageContent(): string | null {
        if (this.rawHistory.length === 0) return null
        return this.rawHistory[this.rawHistory.length - 1].content
    }

    getLastCompactionAttempt(): number {
        return this.lastCompactionAttempt
    }

    // -----------------------------------------------------------------------
    // Mutation helpers (used by VoiceAgentService for cancel / merge)
    // -----------------------------------------------------------------------

    /**
     * Remove and return the last message from raw history.
     * Useful for cancelling a premature assistant response.
     */
    popLastMessage(): TMessage | undefined {
        return this.rawHistory.pop()
    }

    /**
     * Find the last message with the given role, scanning backward.
     */
    getLastMessageByRole(role: string): TMessage | undefined {
        for (let i = this.rawHistory.length - 1; i >= 0; i--) {
            if (this.rawHistory[i].role === role) return this.rawHistory[i]
        }
        return undefined
    }

    /**
     * Update the content of the last message matching `role`.
     * Re-estimates token count automatically. Returns `true` if
     * a matching message was found and updated.
     */
    updateLastMessageByRole(role: string, content: string): boolean {
        for (let i = this.rawHistory.length - 1; i >= 0; i--) {
            if (this.rawHistory[i].role === role) {
                this.rawHistory[i] = {
                    ...this.rawHistory[i],
                    content,
                    tokenCount: this.config.tokenEstimator.estimateTokens(content),
                }
                return true
            }
        }
        return false
    }
}
