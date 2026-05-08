/**
 * Voice Agent Memory Layer with Episodic Compaction
 *
 * Extends the shared BaseMemoryLayer with voice-specific:
 * - Compressed state shape (topics, facts, user preferences)
 * - Compaction patterns (preference detection, fact extraction, dedup)
 * - Tiktoken-based precise token estimation (cl100k_base encoding)
 * - Persistence via injected MemoryStorage adapter
 */

import {
    BaseMemoryLayer,
    type CompactionLLMClient,
    type MemoryLayerConfig,
    type MemoryMessage,
    type MemoryProvenance,
    type MemoryStorage,
    type PersistedMemoryData,
    type TokenEstimator,
} from "./memory-layer"
import { createVoiceTokenEstimator } from "./voice-token-estimator"
import {
    VOICE_COMPACTION_TOKEN_THRESHOLD,
    VOICE_PROTECTED_CONTEXT_TOKENS,
    VOICE_STATE_BUDGET_TOKENS,
    VOICE_SYSTEM_PROMPT_OVERHEAD,
    VOICE_COMPACTION_DEBOUNCE_TURNS,
    COMPACTION_LLM_TEMPERATURE,
    COMPACTION_LLM_MAX_TOKENS,
} from "./config"
import { parseToolResultMessage } from "./tool-helpers"
import {
    buildCompactionEntry,
    canonicalizeCompactionText,
    createEntrySanitizer,
    createMemoryMetadata,
    formatMemoryMetadataTag,
    runStructuredLLMCompaction,
    selectLatestByKey,
    selectLatestMatchingByKey,
    splitCompactionCandidates,
    summarizeCompactionText,
} from "./memory-compaction-utils"

export type VoiceMemoryStorage = MemoryStorage<PersistedVoiceMemory>

// ============================================================================
// Data Models (voice-specific)
// ============================================================================

export interface VoiceMessage extends MemoryMessage {
    /** Cached character count (used for token estimation). */
    charCount: number
}

export interface VoiceMemoryMetadata {
    provenance: MemoryProvenance
}

export interface VoiceTopic extends VoiceMemoryMetadata {
    topic: string
    canonical: string
    turn: number
}

export interface VoiceFact extends VoiceMemoryMetadata {
    summary: string
    canonical: string
    turn: number
}

export interface VoicePreference extends VoiceMemoryMetadata {
    preference: string
    canonical: string
    turn: number
}

export interface VoiceOpenItem extends VoiceMemoryMetadata {
    item: string
    canonical: string
    turn: number
}

export interface VoiceResolvedItem extends VoiceMemoryMetadata {
    item: string
    canonical: string
    resolvedTurn: number
}

export interface VoiceToolResult extends VoiceMemoryMetadata {
    tool: string
    task: string
    canonicalTask: string
    success: boolean
    summary: string
    turn: number
}

export interface CompressedVoiceState {
    conversationTask?: string
    topicsDiscussed: VoiceTopic[]
    keyFacts: VoiceFact[]
    userPreferences: VoicePreference[]
    openItems: VoiceOpenItem[]
    resolvedItems: VoiceResolvedItem[]
    toolResults: VoiceToolResult[]
    lastUpdatedTurn: number
}

/** Shape of the data written to persistent storage. */
export type PersistedVoiceMemory = PersistedMemoryData<CompressedVoiceState, VoiceMessage>

// ============================================================================
// Constants (voice-specific tuning)
// ============================================================================

/** Hard caps for state arrays. */
const MAX_TOPICS = 30
const MAX_KEY_FACTS = 30
const MAX_USER_PREFERENCES = 20
const MAX_OPEN_ITEMS = 15
const MAX_RESOLVED_ITEMS = 20
const MAX_TOOL_RESULTS = 20

const VOICE_STOP_WORDS = new Set([
    "a",
    "an",
    "the",
    "please",
    "just",
    "really",
    "very",
    "about",
    "tell",
    "me",
    "can",
    "you",
    "could",
    "would",
    "like",
    "want",
    "to",
    "for",
    "my",
    "our",
])

const PERSISTED_RULE_PROVENANCE: MemoryProvenance = { source: "rule", detail: "persisted" }
const PERSISTED_ASSISTANT_PROVENANCE: MemoryProvenance = { source: "assistant", detail: "persisted" }
const PERSISTED_USER_PROVENANCE: MemoryProvenance = { source: "user", detail: "persisted" }
const PERSISTED_TOOL_PROVENANCE: MemoryProvenance = { source: "tool", detail: "persisted" }

function canonicalizeVoiceText(text: string): string {
    return canonicalizeCompactionText(text, {
        stopWords: VOICE_STOP_WORDS,
    })
}

function summarizeVoiceText(text: string, max = 150): string {
    return summarizeCompactionText(text, max)
}

function splitVoiceCandidates(text: string): string[] {
    return splitCompactionCandidates(text, { minLength: 8 })
}

function buildVoiceTopic(topic: string, turn: number, provenance: MemoryProvenance): VoiceTopic | null {
    return buildCompactionEntry({
        text: topic,
        provenance,
        preprocess: (value) => value
            .replace(/^(?:tell me|can you explain|could you explain|what is|what are|how does|help me with)\s+/i, "")
            .replace(/^(?:i want to talk about|let'?s talk about)\s+/i, ""),
        summarize: (value) => summarizeVoiceText(value, 80),
        canonicalize: canonicalizeVoiceText,
        build: (normalized, canonical, metadata) => ({
            topic: normalized,
            canonical,
            turn,
            ...metadata,
        }),
    })
}

function buildVoiceFact(summary: string, turn: number, provenance: MemoryProvenance): VoiceFact | null {
    return buildCompactionEntry({
        text: summary,
        provenance,
        summarize: (value) => summarizeVoiceText(value, 140),
        canonicalize: canonicalizeVoiceText,
        build: (normalized, canonical, metadata) => ({
            summary: normalized,
            canonical,
            turn,
            ...metadata,
        }),
    })
}

function buildVoicePreference(preference: string, turn: number, provenance: MemoryProvenance): VoicePreference | null {
    return buildCompactionEntry({
        text: preference,
        provenance,
        preprocess: (value) => value
            .replace(/^(?:i (?:prefer|like|want)|please|my preference is)\s+/i, "")
            .replace(/^(?:don't|do not)\s+/i, "avoid "),
        summarize: (value) => summarizeVoiceText(value, 120),
        canonicalize: canonicalizeVoiceText,
        build: (normalized, canonical, metadata) => ({
            preference: normalized,
            canonical,
            turn,
            ...metadata,
        }),
    })
}

function buildVoiceOpenItem(item: string, turn: number, provenance: MemoryProvenance): VoiceOpenItem | null {
    return buildCompactionEntry({
        text: item,
        provenance,
        summarize: (value) => summarizeVoiceText(value, 140),
        canonicalize: canonicalizeVoiceText,
        build: (normalized, canonical, metadata) => ({
            item: normalized,
            canonical,
            turn,
            ...metadata,
        }),
    })
}

function buildVoiceResolvedItem(item: string, resolvedTurn: number, provenance: MemoryProvenance): VoiceResolvedItem | null {
    return buildCompactionEntry({
        text: item,
        provenance,
        summarize: (value) => summarizeVoiceText(value, 140),
        canonicalize: canonicalizeVoiceText,
        build: (normalized, canonical, metadata) => ({
            item: normalized,
            canonical,
            resolvedTurn,
            ...metadata,
        }),
    })
}

function buildVoiceToolResult(
    tool: string,
    task: string,
    success: boolean,
    summary: string,
    turn: number,
    provenance: MemoryProvenance,
): VoiceToolResult | null {
    const normalizedTask = summarizeVoiceText(task, 100)
    const canonicalTask = canonicalizeVoiceText(normalizedTask)
    const normalizedSummary = summarizeVoiceText(summary, 150)
    if (!tool || !normalizedTask || !normalizedSummary || canonicalTask.length === 0) {
        return null
    }
    return {
        tool,
        task: normalizedTask,
        canonicalTask,
        success,
        summary: normalizedSummary,
        turn,
        ...createMemoryMetadata(provenance),
    }
}

// ============================================================================
// Voice Memory Layer
// ============================================================================

function createInitialVoiceState(): CompressedVoiceState {
    return {
        topicsDiscussed: [],
        keyFacts: [],
        userPreferences: [],
        openItems: [],
        resolvedItems: [],
        toolResults: [],
        lastUpdatedTurn: 0,
    }
}

function extractVoicePreferenceCandidates(text: string): string[] {
    return splitVoiceCandidates(text).filter((candidate) =>
        /\b(?:i (?:prefer|like|want|always|usually|never)|please|don't|do not|my (?:name|language|preference|location|timezone))\b/i.test(candidate),
    )
}

function extractVoiceFactCandidates(text: string): string[] {
    return splitVoiceCandidates(text).filter((candidate) =>
        /\b(?:the (?:answer|result|solution|temperature|price|time|date|name) (?:is|was|will be)|according to|based on|it (?:is|was|will be|appears|seems))\b/i.test(candidate),
    )
}

function extractVoiceOpenItemCandidates(text: string): string[] {
    return splitVoiceCandidates(text).filter((candidate) =>
        /\?$/.test(candidate) || /\b(?:can you|could you|please|help me|remind me|what|when|where|why|how)\b/i.test(candidate),
    )
}

const sanitizeVoiceTopics = createEntrySanitizer<VoiceTopic>({
    defaultProvenance: PERSISTED_RULE_PROVENANCE,
    fromString: (value, context) => buildVoiceTopic(value, context.turn, context.provenance),
    fromObject: (topic, context) => buildVoiceTopic(
        typeof topic.topic === "string" ? topic.topic : "",
        context.turn,
        context.provenance,
    ),
})

const sanitizeVoiceFacts = createEntrySanitizer<VoiceFact>({
    defaultProvenance: PERSISTED_ASSISTANT_PROVENANCE,
    fromString: (value, context) => buildVoiceFact(value, context.turn, context.provenance),
    fromObject: (fact, context) => buildVoiceFact(
        typeof fact.summary === "string" ? fact.summary : "",
        context.turn,
        context.provenance,
    ),
})

const sanitizeVoicePreferences = createEntrySanitizer<VoicePreference>({
    defaultProvenance: PERSISTED_USER_PROVENANCE,
    fromString: (value, context) => buildVoicePreference(value, context.turn, context.provenance),
    fromObject: (preference, context) => buildVoicePreference(
        typeof preference.preference === "string" ? preference.preference : "",
        context.turn,
        context.provenance,
    ),
})

const sanitizeVoiceOpenItems = createEntrySanitizer<VoiceOpenItem>({
    defaultProvenance: PERSISTED_USER_PROVENANCE,
    fromString: (value, context) => buildVoiceOpenItem(value, context.turn, context.provenance),
    fromObject: (item, context) => buildVoiceOpenItem(
        typeof item.item === "string" ? item.item : "",
        context.turn,
        context.provenance,
    ),
})

const sanitizeVoiceResolvedItems = createEntrySanitizer<VoiceResolvedItem>({
    defaultProvenance: PERSISTED_ASSISTANT_PROVENANCE,
    fromObject: (item, context) => buildVoiceResolvedItem(
        typeof item.item === "string" ? item.item : "",
        context.turn,
        context.provenance,
    ),
    getTurn: (item, fallbackTurn) =>
        typeof item.resolvedTurn === "number" && Number.isFinite(item.resolvedTurn)
            ? item.resolvedTurn
            : fallbackTurn,
})

const sanitizeVoiceToolResults = createEntrySanitizer<VoiceToolResult>({
    defaultProvenance: PERSISTED_TOOL_PROVENANCE,
    fromObject: (result, context) => buildVoiceToolResult(
        typeof result.tool === "string" ? result.tool : "",
        typeof result.task === "string" ? result.task : "",
        typeof result.success === "boolean" ? result.success : false,
        typeof result.summary === "string" ? result.summary : "",
        context.turn,
        context.provenance,
    ),
})

export class VoiceMemoryLayer extends BaseMemoryLayer<CompressedVoiceState, VoiceMessage> {
    /** Optional persistent storage adapter. */
    private storage: VoiceMemoryStorage | null = null

    private constructor(
        tokenEstimator: TokenEstimator,
        storage?: VoiceMemoryStorage,
        compactionClient?: CompactionLLMClient,
        configOverrides?: Partial<MemoryLayerConfig>,
    ) {
        super(
            {
                compactionTokenThreshold: VOICE_COMPACTION_TOKEN_THRESHOLD,
                protectedContextTokens: VOICE_PROTECTED_CONTEXT_TOKENS,
                stateBudgetTokens: VOICE_STATE_BUDGET_TOKENS,
                compactionDebounceTurns: VOICE_COMPACTION_DEBOUNCE_TURNS,
                systemPromptOverhead: VOICE_SYSTEM_PROMPT_OVERHEAD,
                tokenEstimator,
                compactionClient,
                compactionTemperature: COMPACTION_LLM_TEMPERATURE,
                compactionMaxTokens: COMPACTION_LLM_MAX_TOKENS,
                ...configOverrides,
            },
            createInitialVoiceState(),
        )
        this.storage = storage ?? null
    }

    static async create(
        storage?: VoiceMemoryStorage,
        compactionClient?: CompactionLLMClient,
        configOverrides?: Partial<MemoryLayerConfig>,
    ): Promise<VoiceMemoryLayer> {
        const estimator = await createVoiceTokenEstimator()
        return new VoiceMemoryLayer(estimator, storage, compactionClient, configOverrides)
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /** Free tiktoken encoder resources. Call when no longer needed. */
    dispose(): void {
        this.config.tokenEstimator.dispose?.()
    }

    // -----------------------------------------------------------------------
    // Abstract Implementation
    // -----------------------------------------------------------------------

    protected createMessage(
        id: string,
        role: "system" | "user" | "assistant" | "tool",
        content: string,
        turn: number,
        timestamp: number,
    ): VoiceMessage {
        return {
            id,
            role,
            content,
            timestamp,
            turn,
            tokenCount: this.config.tokenEstimator.estimateTokens(content),
            charCount: content.length,
        }
    }

    protected performCompaction(messages: VoiceMessage[]): void {
        for (const msg of messages) {
            if (msg.role === "user") {
                const content = msg.content.trim()
                if (content.length > 10) {
                    const topic = buildVoiceTopic(content, msg.turn, {
                        source: "user",
                        detail: "message",
                    })
                    if (topic) {
                        this.compressedState.topicsDiscussed.push(topic)
                    }
                }

                for (const candidate of extractVoicePreferenceCandidates(content)) {
                    const preference = buildVoicePreference(candidate, msg.turn, {
                        source: "user",
                        detail: "message",
                    })
                    if (preference) {
                        this.compressedState.userPreferences.push(preference)
                    }
                }

                for (const candidate of extractVoiceOpenItemCandidates(content)) {
                    const item = buildVoiceOpenItem(candidate, msg.turn, {
                        source: "user",
                        detail: "message",
                    })
                    if (item) {
                        this.compressedState.openItems.push(item)
                    }
                }
            }

            if (msg.role === "assistant") {
                const content = msg.content.trim()
                for (const candidate of extractVoiceFactCandidates(content)) {
                    const fact = buildVoiceFact(candidate, msg.turn, {
                        source: "assistant",
                        detail: "message",
                    })
                    if (fact) {
                        this.compressedState.keyFacts.push(fact)
                    }
                }

                if (/\b(?:done|resolved|finished|completed)\b/i.test(content)) {
                    const resolved = buildVoiceResolvedItem(content, msg.turn, {
                        source: "assistant",
                        detail: "message",
                    })
                    if (resolved) {
                        this.compressedState.resolvedItems.push(resolved)
                    }
                }
            }

            if (msg.role === "tool") {
                const toolInfo = parseToolResultMessage(msg.content)
                if (toolInfo.toolName && toolInfo.parsed) {
                    const summarySource = toolInfo.parsed.output ?? toolInfo.parsed.stderr ?? toolInfo.error ?? ""
                    const summary = typeof summarySource === "string"
                        ? summarySource.slice(0, 150)
                        : JSON.stringify(summarySource).slice(0, 150)
                    const toolResult = buildVoiceToolResult(
                        toolInfo.toolName,
                        typeof toolInfo.parsed.command === "string" && toolInfo.parsed.command.length > 0
                            ? toolInfo.parsed.command
                            : toolInfo.toolName,
                        toolInfo.success,
                        summary,
                        msg.turn,
                        { source: "tool", detail: toolInfo.toolName },
                    )
                    if (toolResult) {
                        this.compressedState.toolResults.push(toolResult)
                    }
                }
            }
        }

        this.compressedState.lastUpdatedTurn = this.currentTurn
        this.finalizeCompactedState()
    }

    protected override async performLLMCompaction(messages: VoiceMessage[]): Promise<boolean> {
        const parsed = await runStructuredLLMCompaction({
            client: this.config.compactionClient,
            currentState: this.renderState(),
            messages,
            formatMessage: (message) => `[${message.role} turn=${message.turn}] ${message.content}`,
            persona: "You are a conversation memory compactor. Given the conversation messages and current compacted state, produce an UPDATED compacted state as JSON.",
            schema: '{"conversationTask":"string or null","topicsDiscussed":[{"topic":"string (max 80 chars)","canonical":"string","turn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}],"keyFacts":[{"summary":"string","canonical":"string","turn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}],"userPreferences":[{"preference":"string","canonical":"string","turn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}],"openItems":[{"item":"string","canonical":"string","turn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}],"resolvedItems":[{"item":"string","canonical":"string","resolvedTurn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}],"toolResults":[{"tool":"string","task":"string","canonicalTask":"string","success":boolean,"summary":"string (max 150 chars)","turn":number,"provenance":{"source":"rule|llm|tool|assistant|user","detail":"string or null"}}]}',
            rules: [
                "Merge new information with existing state — do NOT discard existing entries unless superseded",
                "Deduplicate similar topics and facts",
                "Normalize topics, preferences, and tasks into stable canonical strings so paraphrases merge",
                "Keep entries concise",
                "Return ONLY valid JSON — no markdown fences, no commentary",
            ],
            temperature: this.config.compactionTemperature,
            maxTokens: this.config.compactionMaxTokens,
            parseState: (obj) => this.parseLLMCompactionObject(obj),
        })
        if (parsed) {
            this.compressedState = parsed
            this.finalizeCompactedState()
            return true
        }
        return false
    }

    private parseLLMCompactionObject(obj: Record<string, unknown>): CompressedVoiceState | null {
        return {
            conversationTask: typeof obj.conversationTask === "string"
                ? obj.conversationTask
                : this.compressedState.conversationTask,
            topicsDiscussed: sanitizeVoiceTopics(obj.topicsDiscussed, this.currentTurn, { source: "llm", detail: "compaction" }),
            keyFacts: sanitizeVoiceFacts(obj.keyFacts, this.currentTurn, { source: "llm", detail: "compaction" }),
            userPreferences: sanitizeVoicePreferences(obj.userPreferences, this.currentTurn, { source: "llm", detail: "compaction" }),
            openItems: sanitizeVoiceOpenItems(obj.openItems, this.currentTurn, { source: "llm", detail: "compaction" }),
            resolvedItems: sanitizeVoiceResolvedItems(obj.resolvedItems, this.currentTurn, { source: "llm", detail: "compaction" }),
            toolResults: sanitizeVoiceToolResults(obj.toolResults, this.currentTurn, { source: "llm", detail: "compaction" }),
            lastUpdatedTurn: this.currentTurn,
        }
    }

    private finalizeCompactedState(): void {
        // 1. Remove old resolved items (older than 15 turns)
        this.compressedState.resolvedItems = selectLatestMatchingByKey(
            this.compressedState.resolvedItems,
            {
                keyOf: (item) => item.canonical,
                recencyOf: (item) => item.resolvedTurn,
                include: (item) => this.currentTurn - item.resolvedTurn < 15,
            },
        )

        this.compressedState.keyFacts = selectLatestByKey(
            this.compressedState.keyFacts,
            (fact) => fact.canonical,
            (fact) => fact.turn,
        )

        this.compressedState.userPreferences = selectLatestByKey(
            this.compressedState.userPreferences,
            (preference) => preference.canonical,
            (preference) => preference.turn,
        )

        // 3. Deduplicate topics by content (keep latest turn)
        this.compressedState.topicsDiscussed = selectLatestByKey(
            this.compressedState.topicsDiscussed,
            (topic) => topic.canonical,
            (topic) => topic.turn,
        )

        this.compressedState.openItems = selectLatestMatchingByKey(
            this.compressedState.openItems,
            {
                keyOf: (item) => item.canonical,
                recencyOf: (item) => item.turn,
                include: (item) => !this.compressedState.resolvedItems.some(
                    (resolved) => resolved.canonical === item.canonical,
                ),
            },
        )

        // 4. Age out old tool results and dedup by tool+summary
        this.compressedState.toolResults =
            this.compressedState.toolResults.filter(
                (tr) => this.currentTurn - tr.turn <= 20,
            )
        this.compressedState.toolResults = selectLatestMatchingByKey(
            this.compressedState.toolResults,
            {
                keyOf: (toolResult) => `${toolResult.tool}:${toolResult.canonicalTask}:${toolResult.summary.substring(0, 60).toLowerCase()}`,
                recencyOf: (toolResult) => toolResult.turn,
                include: (toolResult) => this.currentTurn - toolResult.turn <= 20,
            },
        )

        // 5. Enforce hard caps (keep most recent)
        if (this.compressedState.topicsDiscussed.length > MAX_TOPICS) {
            this.compressedState.topicsDiscussed =
                this.compressedState.topicsDiscussed.slice(-MAX_TOPICS)
        }
        if (this.compressedState.keyFacts.length > MAX_KEY_FACTS) {
            this.compressedState.keyFacts =
                this.compressedState.keyFacts.slice(-MAX_KEY_FACTS)
        }
        if (this.compressedState.userPreferences.length > MAX_USER_PREFERENCES) {
            this.compressedState.userPreferences =
                this.compressedState.userPreferences.slice(-MAX_USER_PREFERENCES)
        }
        if (this.compressedState.openItems.length > MAX_OPEN_ITEMS) {
            this.compressedState.openItems =
                this.compressedState.openItems.slice(-MAX_OPEN_ITEMS)
        }
        if (this.compressedState.resolvedItems.length > MAX_RESOLVED_ITEMS) {
            this.compressedState.resolvedItems =
                this.compressedState.resolvedItems.slice(-MAX_RESOLVED_ITEMS)
        }
        if (this.compressedState.toolResults.length > MAX_TOOL_RESULTS) {
            this.compressedState.toolResults =
                this.compressedState.toolResults.slice(-MAX_TOOL_RESULTS)
        }
    }

    protected renderState(): string {
        const parts: string[] = []

        if (this.compressedState.conversationTask) {
            parts.push(`**Task**: ${this.compressedState.conversationTask}`)
        }

        if (this.compressedState.topicsDiscussed.length > 0) {
            const topics = this.compressedState.topicsDiscussed
                .map((t) => `- [Turn ${t.turn}] ${t.topic}${formatMemoryMetadataTag(t)}`)
                .join("\n")
            parts.push(`**Topics Discussed**:\n${topics}`)
        }

        if (this.compressedState.keyFacts.length > 0) {
            parts.push(`**Key Facts**:\n${this.compressedState.keyFacts
                .map((fact) => `- [Turn ${fact.turn}] ${fact.summary}${formatMemoryMetadataTag(fact)}`)
                .join("\n")}`)
        }

        if (this.compressedState.userPreferences.length > 0) {
            parts.push(
                `**User Preferences**:\n${this.compressedState.userPreferences
                    .map((preference) => `- [Turn ${preference.turn}] ${preference.preference}${formatMemoryMetadataTag(preference)}`)
                    .join("\n")}`,
            )
        }

        if (this.compressedState.openItems.length > 0) {
            parts.push(
                `**Open Items**:\n${this.compressedState.openItems
                    .map((item) => `- [Turn ${item.turn}] ${item.item}${formatMemoryMetadataTag(item)}`)
                    .join("\n")}`,
            )
        }

        if (this.compressedState.resolvedItems.length > 0) {
            const resolved = this.compressedState.resolvedItems
                .map((r) => `${r.item} (turn ${r.resolvedTurn})${formatMemoryMetadataTag(r)}`)
                .join(", ")
            parts.push(`**Resolved**: ${resolved}`)
        }

        if (this.compressedState.toolResults.length > 0) {
            const tools = this.compressedState.toolResults
                .map(
                    (t) =>
                        `- [Turn ${t.turn}] ${t.tool}: ${t.success ? "✓" : "✗"} ${t.summary}${formatMemoryMetadataTag(t)}`,
                )
                .join("\n")
            parts.push(`**Tool Results**:\n${tools}`)
        }

        return parts.join("\n\n")
    }

    protected override createInitialState(): CompressedVoiceState {
        return createInitialVoiceState()
    }

    protected override getStateLabel(): string {
        return "Conversation Memory"
    }

    protected override getResumePlaceholder(): string {
        return "Continue our conversation."
    }

    // -----------------------------------------------------------------------
    // Override: keep charCount in sync when content is updated
    // -----------------------------------------------------------------------

    override updateLastMessageByRole(role: string, content: string): boolean {
        for (let i = this.rawHistory.length - 1; i >= 0; i--) {
            if (this.rawHistory[i].role === role) {
                this.rawHistory[i] = {
                    ...this.rawHistory[i],
                    content,
                    tokenCount: this.config.tokenEstimator.estimateTokens(content),
                    charCount: content.length,
                }
                return true
            }
        }
        return false
    }

    /** Merge adjacent messages of the same role into a single raw-history entry. */
    coalesceConsecutiveMessages(
        role: VoiceMessage["role"],
        separator = "\n\n",
    ): boolean {
        if (this.rawHistory.length < 2) {
            return false
        }

        const merged: VoiceMessage[] = []
        let changed = false

        for (const message of this.rawHistory) {
            const last = merged[merged.length - 1]
            if (last && last.role === role && message.role === role) {
                const content = `${last.content}${separator}${message.content}`
                merged[merged.length - 1] = {
                    ...last,
                    content,
                    tokenCount: this.config.tokenEstimator.estimateTokens(content),
                    charCount: content.length,
                }
                changed = true
                continue
            }

            merged.push(message)
        }

        if (!changed) {
            return false
        }

        this.rawHistory = merged
        return true
    }

    /** Rewrite matching raw-history messages in place while keeping cached counts in sync. */
    rewriteMessages(
        role: VoiceMessage["role"],
        rewrite: (message: VoiceMessage) => string | null | undefined,
    ): boolean {
        let changed = false

        this.rawHistory = this.rawHistory.map((message) => {
            if (message.role !== role) {
                return message
            }

            const nextContent = rewrite(message)
            if (typeof nextContent !== "string" || nextContent === message.content) {
                return message
            }

            changed = true
            return {
                ...message,
                content: nextContent,
                tokenCount: this.config.tokenEstimator.estimateTokens(nextContent),
                charCount: nextContent.length,
            }
        })

        return changed
    }

    // -----------------------------------------------------------------------
    // Persistence (delegates to injected storage adapter)
    // -----------------------------------------------------------------------

    /** Load persisted memory from storage (if available). */
    async load(): Promise<void> {
        if (!this.storage) return
        const data = await this.storage.load()
        if (data) {
            this.loadFromPersisted(data)
        }
    }

    /** Persist current memory state to storage (if available). */
    async saveStrict(): Promise<void> {
        if (!this.storage) return
        await this.storage.save(this.serializeBase() as PersistedVoiceMemory)
    }

    /** Persist current memory state to storage (if available). */
    async save(): Promise<void> {
        if (!this.storage) return
        try {
            await this.saveStrict()
        } catch (err) {
            console.error("[VoiceMemory] Failed to save:", err)
        }
    }

    /** Clear all memory (in-memory and persisted). */
    async clear(): Promise<void> {
        this.reset()
        if (this.storage) {
            await this.storage.clear()
        }
    }

    /** Called after compaction to persist. */
    protected override onCompactionComplete(): void {
        this.save().catch(() => {})
    }

    /** Load from persisted data with defensive validation. */
    private loadFromPersisted(data: PersistedVoiceMemory): void {
        if (!Array.isArray(data.rawHistory)) {
            console.warn("[VoiceMemory] Invalid persisted data — starting fresh")
            return
        }

        // Sanitize base fields + voice-specific charCount
        const sanitizedHistory = this.sanitizeHistory(data.rawHistory).map(
            (msg) => ({
                ...msg,
                tokenCount: this.config.tokenEstimator.estimateTokens(msg.content ?? ""),
                charCount: (msg.content ?? "").length,
            }),
        )

        const cs = data.compressedState
        const sanitizedState: CompressedVoiceState = {
            conversationTask: typeof cs?.conversationTask === "string" ? cs.conversationTask : undefined,
            topicsDiscussed: sanitizeVoiceTopics(cs?.topicsDiscussed, 0),
            keyFacts: sanitizeVoiceFacts(cs?.keyFacts, 0),
            userPreferences: sanitizeVoicePreferences(cs?.userPreferences, 0),
            openItems: sanitizeVoiceOpenItems(cs?.openItems, 0),
            resolvedItems: sanitizeVoiceResolvedItems(cs?.resolvedItems, 0),
            toolResults: sanitizeVoiceToolResults(cs?.toolResults, 0),
            lastUpdatedTurn: typeof cs?.lastUpdatedTurn === "number" ? cs.lastUpdatedTurn : 0,
        }

        this.loadBase({
            rawHistory: sanitizedHistory,
            compressedState: sanitizedState,
            compactionCursor: data.compactionCursor,
            currentTurn: data.currentTurn,
            compactionCount: data.compactionCount,
        })

        console.log(
            `[VoiceMemory] Loaded — ${this.getRawHistoryLength()} messages, ` +
                `turn ${this.getCurrentTurn()}, ` +
                `${this.getCompactionCount()} compactions`,
        )
    }
}
