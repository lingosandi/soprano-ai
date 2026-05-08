import {
    createEmptyToolRegistry,
    type ToolRegistry,
} from "./tool-registry"

import { toError } from "./audio-helpers"
import { LLMClient } from "./llm-client"
import type { ChatMessage, FetchLike, LLMProviderConfig } from "./types"
import type { AgentState } from "./voice-agent-types"
import { createVoiceAgentStateActor, STATE_TO_EVENT } from "./voice-agent-state"
import { VoiceMemoryLayer } from "./voice-memory"
import type { VoiceMemoryStorage } from "./voice-memory"

export interface SharedAgentRuntimeDeps {
    llmProviderConfig: LLMProviderConfig
    compactionProviderConfig?: LLMProviderConfig | null
    fetchImpl?: FetchLike
    toolRegistry?: ToolRegistry
    memory?: VoiceMemoryStorage | null
    onStateChange?: (state: AgentState) => void
    onLog?: (message: string) => void
}

export interface SharedAgentMemoryLoadOptions {
    logMessage?: string
    staticContext?: string
    normalizeLoadedMemory?: (
        memory: VoiceMemoryLayer,
    ) => Promise<boolean> | boolean
    afterLoad?: (memory: VoiceMemoryLayer) => Promise<void> | void
}

export interface SharedAgentBuildMessagesOptions {
    systemPrompt: string
    userPrompt?: string | null
    fallbackHistory?: ChatMessage[] | null
    coalesceAdjacentRoles?: boolean
}

export interface SharedAgentMemorySyncOptions {
    swallowError?: boolean
    afterSync?: (memory: VoiceMemoryLayer) => Promise<void> | void
}

export interface SharedAgentAppendMessageOptions
    extends SharedAgentMemorySyncOptions {
    role: ChatMessage["role"]
    content: string
    reason: string
    immediateSync?: boolean
    fallbackHistory?: ChatMessage[] | null
    mergeConsecutiveRole?: ChatMessage["role"] | false
}

export class SharedAgentRuntime {
    readonly llm: LLMClient
    readonly toolRegistry: ToolRegistry

    private actor: ReturnType<typeof createVoiceAgentStateActor>["actor"]
    private stopActor: () => void
    private memoryLayer: VoiceMemoryLayer | null = null
    private deferredMemoryStorage: VoiceMemoryStorage | null
    private deferredFetchImpl?: FetchLike
    private deferredCompactionProviderConfig?: LLMProviderConfig | null
    private onLog?: (message: string) => void
    private memoryReadyPromise: Promise<void> | null = null
    private memoryLoaded = false
    private memorySyncPromise: Promise<void> = Promise.resolve()

    constructor(deps: SharedAgentRuntimeDeps) {
        const providerConfig = { ...deps.llmProviderConfig }
        if (deps.fetchImpl) {
            providerConfig.fetchImpl = deps.fetchImpl
        }

        this.llm = new LLMClient(providerConfig)
        this.toolRegistry = deps.toolRegistry ?? createEmptyToolRegistry()
        this.deferredMemoryStorage = deps.memory ?? null
        this.deferredFetchImpl = deps.fetchImpl
        this.deferredCompactionProviderConfig = deps.compactionProviderConfig ?? deps.llmProviderConfig
        this.onLog = deps.onLog

        const { actor, stop } = createVoiceAgentStateActor(deps.onStateChange)
        this.actor = actor
        this.stopActor = stop
    }

    get memory(): VoiceMemoryLayer | null {
        return this.memoryLayer
    }

    get state(): AgentState {
        return this.actor.getSnapshot().value as AgentState
    }

    get continuousMode(): boolean {
        return this.actor.getSnapshot().context.continuousMode
    }

    set continuousMode(value: boolean) {
        this.actor.send({ type: "SET_CONTINUOUS", value })
    }

    setState(state: AgentState): void {
        this.actor.send({ type: STATE_TO_EVENT[state] })
    }

    destroy(): void {
        const memory = this.memoryLayer
        if (memory) {
            void this.memorySyncPromise.finally(() => memory.dispose())
        }
        this.stopActor()
    }

    async ensureMemoryReady(): Promise<void> {
        if (this.memoryLayer || !this.deferredMemoryStorage) {
            return
        }

        if (this.memoryReadyPromise) {
            return this.memoryReadyPromise
        }

        const promise = (async () => {
            const compactionProvider = this.deferredCompactionProviderConfig
            if (!compactionProvider) {
                throw new Error("Voice memory requires compactionProviderConfig")
            }

            const compactionConfig = { ...compactionProvider }
            if (this.deferredFetchImpl) {
                compactionConfig.fetchImpl = this.deferredFetchImpl
            }

            const compactionClient = new LLMClient(compactionConfig)
            this.memoryLayer = await VoiceMemoryLayer.create(
                this.deferredMemoryStorage!,
                compactionClient,
            )
            this.deferredMemoryStorage = null
        })()

        this.memoryReadyPromise = promise.catch((error) => {
            this.memoryReadyPromise = null
            throw error
        })

        return this.memoryReadyPromise
    }

    async loadMemoryOnce(
        options: SharedAgentMemoryLoadOptions = {},
    ): Promise<void> {
        await this.ensureMemoryReady()
        if (!this.memoryLayer || this.memoryLoaded) {
            return
        }

        await this.memoryLayer.load()

        let changed = false
        if (options.normalizeLoadedMemory) {
            changed = Boolean(await options.normalizeLoadedMemory(this.memoryLayer)) || changed
        }

        if (options.staticContext && !this.memoryLayer.hasStaticContext()) {
            this.memoryLayer.setStaticContext(options.staticContext)
            changed = true
        }

        if (changed) {
            await this.memoryLayer.saveStrict()
        }

        if (options.afterLoad) {
            await options.afterLoad(this.memoryLayer)
        }

        this.memoryLoaded = true
        if (options.logMessage) {
            this.onLog?.(options.logMessage)
        }
    }

    buildMessages(options: SharedAgentBuildMessagesOptions): ChatMessage[] {
        const {
            systemPrompt,
            userPrompt,
            fallbackHistory,
            coalesceAdjacentRoles = true,
        } = options

        const messages = this.memoryLayer
            ? [
                ...this.memoryLayer.buildMessages(systemPrompt),
                ...(userPrompt ? [{ role: "user", content: userPrompt } satisfies ChatMessage] : []),
            ]
            : this.buildFallbackMessages(systemPrompt, userPrompt, fallbackHistory)

        return coalesceAdjacentRoles
            ? this.coalesceAdjacentMessages(messages)
            : messages
    }

    getLastMessageRole(fallbackHistory?: ChatMessage[] | null): string | undefined {
        if (this.memoryLayer) {
            return this.memoryLayer.getLastMessageRole() ?? undefined
        }

        return fallbackHistory?.[fallbackHistory.length - 1]?.role
    }

    getLastMessageContent(fallbackHistory?: ChatMessage[] | null): string | undefined {
        if (this.memoryLayer) {
            return this.memoryLayer.getLastMessageContent() ?? undefined
        }

        return fallbackHistory?.[fallbackHistory.length - 1]?.content
    }

    queueMemorySync(
        reason: string,
        options: SharedAgentMemorySyncOptions = {},
    ): Promise<void> {
        if (!this.memoryLayer) {
            return Promise.resolve()
        }

        const next = this.memorySyncPromise.catch(() => {}).then(async () => {
            if (!this.memoryLayer) {
                return
            }

            await this.memoryLayer.triggerCompactionIfNeeded()
            await this.memoryLayer.saveStrict()
            if (options.afterSync) {
                await options.afterSync(this.memoryLayer)
            }
        })

        this.memorySyncPromise = next.catch(() => {})

        const handled = next.catch((error) => {
            this.onLog?.(`${reason}: ${toError(error).message}`)
            throw error
        })

        return options.swallowError
            ? handled.catch(() => {})
            : handled
    }

    scheduleMemorySync(
        reason: string,
        options: Omit<SharedAgentMemorySyncOptions, "swallowError"> = {},
    ): void {
        void this.queueMemorySync(reason, {
            ...options,
            swallowError: true,
        })
    }

    async appendHistoryMessage(
        options: SharedAgentAppendMessageOptions,
    ): Promise<void> {
        const {
            role,
            content,
            reason,
            immediateSync = false,
            fallbackHistory,
            mergeConsecutiveRole = false,
            afterSync,
        } = options

        const nextContent = role === "user" ? content.trim() : content
        if (!nextContent) {
            return
        }

        if (this.memoryLayer) {
            if (
                mergeConsecutiveRole &&
                role === mergeConsecutiveRole &&
                this.memoryLayer.getLastMessageRole() === role
            ) {
                const lastContent = this.memoryLayer.getLastMessageContent() ?? ""
                this.memoryLayer.updateLastMessageByRole(
                    role,
                    `${lastContent}\n\n${nextContent}`,
                )
            } else {
                this.memoryLayer.appendMessage(role, nextContent)
            }

            if (immediateSync) {
                await this.queueMemorySync(reason, {
                    afterSync,
                    swallowError: true,
                })
            } else {
                this.scheduleMemorySync(reason, { afterSync })
            }
            return
        }

        if (!fallbackHistory) {
            return
        }

        const lastMessage = fallbackHistory[fallbackHistory.length - 1]
        if (
            mergeConsecutiveRole &&
            role === mergeConsecutiveRole &&
            lastMessage?.role === role
        ) {
            lastMessage.content = `${lastMessage.content}\n\n${nextContent}`
            return
        }

        fallbackHistory.push({ role, content: nextContent })
    }

    private buildFallbackMessages(
        systemPrompt: string,
        userPrompt?: string | null,
        fallbackHistory?: ChatMessage[] | null,
    ): ChatMessage[] {
        const messages = (fallbackHistory ?? []).map((message) => ({ ...message }))

        if (messages.length === 0 || messages[0].role !== "system") {
            messages.unshift({ role: "system", content: systemPrompt })
        } else {
            messages[0] = { ...messages[0], content: systemPrompt }
        }

        if (userPrompt) {
            messages.push({ role: "user", content: userPrompt })
        }

        return messages
    }

    private coalesceAdjacentMessages(messages: ChatMessage[]): ChatMessage[] {
        const coalesced: ChatMessage[] = []

        for (const message of messages) {
            const lastMessage = coalesced[coalesced.length - 1]
            if (lastMessage && lastMessage.role === message.role) {
                lastMessage.content += `\n\n${message.content}`
                continue
            }

            coalesced.push({ ...message })
        }

        return coalesced
    }
}