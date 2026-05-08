import type { ToolRegistry } from "./tool-registry"

import {
    SharedAgentRuntime,
    type SharedAgentAppendMessageOptions,
    type SharedAgentBuildMessagesOptions,
    type SharedAgentMemoryLoadOptions,
    type SharedAgentMemorySyncOptions,
} from "./shared-agent-runtime"
import type {
    ChatMessage,
    FetchLike,
    LLMProviderConfig,
    ProviderName,
    ToolDefinition,
} from "./types"
import type { AgentState, VoiceAgentCallbacks } from "./voice-agent-types"
import { VoiceMemoryLayer } from "./voice-memory"
import type { VoiceMemoryStorage } from "./voice-memory"

export interface ToolCallingAgentBaseDeps {
    provider?: ProviderName
    llmProviderConfig: LLMProviderConfig
    compactionProviderConfig?: LLMProviderConfig | null
    fetchImpl?: FetchLike
    toolRegistry?: ToolRegistry
    memory?: VoiceMemoryStorage | null
}

export abstract class ToolCallingAgentBase {
    private registeredCallbacks: VoiceAgentCallbacks = {}

    protected readonly runtime: SharedAgentRuntime

    protected constructor(deps: ToolCallingAgentBaseDeps) {
        this.runtime = new SharedAgentRuntime({
            llmProviderConfig: deps.llmProviderConfig,
            compactionProviderConfig: deps.compactionProviderConfig,
            fetchImpl: deps.fetchImpl,
            toolRegistry: deps.toolRegistry,
            memory: deps.memory ?? null,
            onLog: (message) => this.callbacks.onLog?.(message),
        })
    }

    on(cb: VoiceAgentCallbacks): void {
        this.registeredCallbacks = { ...this.registeredCallbacks, ...cb }
    }

    get callbackHandlers(): VoiceAgentCallbacks {
        return this.registeredCallbacks
    }

    get state(): AgentState {
        return this.runtime.state
    }

    protected get callbacks(): VoiceAgentCallbacks {
        return this.registeredCallbacks
    }

    protected get memory(): VoiceMemoryLayer | null {
        return this.runtime.memory
    }

    protected get toolRegistry(): ToolRegistry {
        return this.runtime.toolRegistry
    }

    protected get continuousMode(): boolean {
        return this.runtime.continuousMode
    }

    protected set continuousMode(value: boolean) {
        this.runtime.continuousMode = value
    }

    protected setAgentState(state: AgentState): void {
        this.runtime.setState(state)
        this.callbacks.onStateChange?.(this.runtime.state)
    }

    protected getRegisteredToolDefinitions(): ToolDefinition[] {
        return this.toolRegistry.getToolDefinitions()
    }

    protected getExecutableToolDefinitions(): ToolDefinition[] {
        const canExecute = this.toolRegistry.canExecute?.() ?? this.toolRegistry.getToolDefinitions().length > 0
        if (!canExecute) {
            return []
        }

        return this.toolRegistry.getToolDefinitions()
    }

    protected ensureMemoryReady(): Promise<void> {
        return this.runtime.ensureMemoryReady()
    }

    protected loadMemoryOnce(
        options: SharedAgentMemoryLoadOptions = {},
    ): Promise<void> {
        return this.runtime.loadMemoryOnce(options)
    }

    protected buildMessages(
        options: SharedAgentBuildMessagesOptions,
    ): ChatMessage[] {
        return this.runtime.buildMessages(options)
    }

    protected appendHistoryMessage(
        options: SharedAgentAppendMessageOptions,
    ): Promise<void> {
        return this.runtime.appendHistoryMessage(options)
    }

    protected queueMemorySync(
        reason: string,
        options: SharedAgentMemorySyncOptions = {},
    ): Promise<void> {
        return this.runtime.queueMemorySync(reason, options)
    }

    protected scheduleMemorySync(
        reason: string,
        options: Omit<SharedAgentMemorySyncOptions, "swallowError"> = {},
    ): void {
        this.runtime.scheduleMemorySync(reason, options)
    }

    protected getLastMessageRole(
        fallbackHistory?: ChatMessage[] | null,
    ): string | undefined {
        return this.runtime.getLastMessageRole(fallbackHistory)
    }

    protected getLastMessageContent(
        fallbackHistory?: ChatMessage[] | null,
    ): string | undefined {
        return this.runtime.getLastMessageContent(fallbackHistory)
    }

    protected destroyRuntime(): void {
        this.runtime.destroy()
    }
}