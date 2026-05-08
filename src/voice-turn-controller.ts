import { AgentTurnOrchestrator } from "./agent-turn-orchestrator"
import { toError } from "./audio-helpers"
import type { SharedAgentRuntime } from "./shared-agent-runtime"
import {
    StreamingToolTurnRunner,
    type StreamingToolTurnRunOptions,
    type StreamingToolTurnResult,
} from "./streaming-tool-turn-runner"
import { ToolCallExecutionRunner } from "./tool-call-execution-runner"
import type { RunInvocationTarget } from "./command-response-utils"
import type { ChatMessage, ToolCall } from "./types"
import { VoiceValidationEnforcer } from "./voice-validation-enforcer"

export interface VoiceTurnControllerRunOptions {
    text: string
    skipHistoryPush: boolean
    skipSpeech: boolean
    signal: AbortSignal
}

export interface VoiceTurnControllerDeps {
    runtime: SharedAgentRuntime
    enforcer: VoiceValidationEnforcer
    getFallbackHistory: () => ChatMessage[]
    getSystemPrompt: () => string
    temperature: number
    maxTokens: number
    requestExtraBody?: Record<string, unknown>
    allowedRunTargets?: readonly RunInvocationTarget[]
    speechBufferThreshold: number
    sentenceEndPattern: RegExp
    sendSpeechChunk: (contextId: string, chunk: string) => Promise<void> | void
    flushSpeech: (contextId: string) => Promise<void> | void
    onStreamStart?: () => void
    onFirstSpeechToken?: (options: { skipSpeech: boolean }) => void
    onPartialResponse?: (content: string) => void
    onToolCall?: (selectedToolCall: ToolCall) => void
    onToolCallError?: (error: Error, rawCommandText: string) => void
    onFullResponse?: (content: string) => void
    onStreamError?: (error: Error) => Promise<void> | void
    abortCurrentTurn?: () => void
    onLog?: (message: string) => void
    onError?: (error: Error) => void
    appendToolMessage: (content: string) => Promise<void>
    onToolStart?: (
        name: string,
        parameters: Record<string, unknown>,
        index: number,
        total: number,
    ) => void
    beforeTurn?: (context: { isContinuation: boolean }) => Promise<void> | void
    createTurnOptions: (context: {
        runOptions: VoiceTurnControllerRunOptions
        isContinuation: boolean
        previousOptions?: StreamingToolTurnRunOptions
        previousTurnResult?: StreamingToolTurnResult
    }) => StreamingToolTurnRunOptions
    afterToolExecution?: (context: {
        runOptions: VoiceTurnControllerRunOptions
        loopOptions: StreamingToolTurnRunOptions
        turnResult: StreamingToolTurnResult
        selectedToolCall: ToolCall
        iteration: number
    }) => Promise<{ aborted?: boolean } | void> | { aborted?: boolean } | void
    finalizeResult?: (context: {
        runOptions: VoiceTurnControllerRunOptions
        finalTurnResult: StreamingToolTurnResult
    }) => Promise<void> | void
    restoreAfterError: () => Promise<void> | void
}

export class VoiceTurnController {
    private turnOrchestrator: AgentTurnOrchestrator<
        VoiceTurnControllerRunOptions,
        StreamingToolTurnRunOptions,
        StreamingToolTurnResult,
        void
    >

    constructor(private deps: VoiceTurnControllerDeps) {
        const turnRunner = new StreamingToolTurnRunner({
            runtime: deps.runtime,
            llm: deps.runtime.llm,
            getSystemPrompt: () => deps.getSystemPrompt(),
            getFallbackHistory: () => deps.getFallbackHistory(),
            temperature: deps.temperature,
            maxTokens: deps.maxTokens,
            requestExtraBody: deps.requestExtraBody,
            allowedRunTargets: deps.allowedRunTargets,
            speechBufferThreshold: deps.speechBufferThreshold,
            sentenceEndPattern: deps.sentenceEndPattern,
            speechSink: {
                sendSpeechChunk: (contextId, chunk) =>
                    deps.sendSpeechChunk(contextId, chunk),
                flushSpeech: async (contextId) => {
                    await deps.flushSpeech(contextId)
                },
            },
            onStreamStart: () => {
                deps.onStreamStart?.()
            },
            onFirstSpeechToken: (options) => {
                deps.onFirstSpeechToken?.(options)
            },
            onPartialResponse: (content) => {
                deps.onPartialResponse?.(content)
            },
            onToolCall: (selectedToolCall) => {
                deps.onToolCall?.(selectedToolCall)
            },
            onToolCallError: (error, rawCommandText) => {
                deps.onToolCallError?.(error, rawCommandText)
            },
            handleToolCallParseError: async (error, rawCommandText) => {
                await deps.enforcer.handleToolCallParseError(error, rawCommandText)
            },
            onFullResponse: (content) => {
                deps.onFullResponse?.(content)
            },
            onStreamError: async (error) => {
                await deps.onStreamError?.(error)
            },
            abortCurrentTurn: () => {
                deps.abortCurrentTurn?.()
            },
            onLog: (message) => deps.onLog?.(message),
            onError: (error) => deps.onError?.(error),
        })

        const toolExecutionRunner = new ToolCallExecutionRunner({
            toolRegistry: deps.runtime.toolRegistry,
            appendToolMessage: async (content) => {
                await deps.appendToolMessage(content)
            },
            onToolStart: (name, parameters, index, total) => {
                deps.onToolStart?.(name, parameters, index, total)
            },
            onError: (error) => deps.onError?.(error),
            onLog: (message) => deps.onLog?.(message),
        })

        this.turnOrchestrator = new AgentTurnOrchestrator({
            turnRunner,
            getSelectedToolCall: (turnResult) => turnResult.pendingSelectedToolCall,
            isAborted: ({ runOptions, turnResult }) =>
                turnResult.aborted || runOptions.signal.aborted,
            executeSelectedToolCall: async ({ runOptions, selectedToolCall }) =>
                toolExecutionRunner.run({
                    selectedToolCall,
                    signal: runOptions.signal,
                }),
            afterToolExecution: deps.afterToolExecution
                ? async (context) => deps.afterToolExecution?.(context)
                : undefined,
            beforeTurn: async ({ isContinuation }) => {
                await deps.beforeTurn?.({ isContinuation })
            },
            createTurnOptions: ({
                runOptions,
                isContinuation,
                previousOptions,
                previousTurnResult,
            }) =>
                deps.createTurnOptions({
                    runOptions,
                    isContinuation,
                    previousOptions,
                    previousTurnResult,
                }),
            finalizeResult: async (context) => {
                await deps.finalizeResult?.(context)
            },
            loopLabel: "Voice tool-calling",
            onLog: (message) => deps.onLog?.(message),
        })
    }

    async run(runOptions: VoiceTurnControllerRunOptions): Promise<void> {
        try {
            await this.turnOrchestrator.run(runOptions)
        } catch (error) {
            const nextError = toError(error)
            if (runOptions.signal.aborted || isAbortError(nextError)) {
                this.deps.onLog?.("LLM turn aborted")
                return
            }
            this.deps.onLog?.(`LLM catch error: ${nextError.message}`)
            this.deps.onError?.(nextError)
            await this.deps.enforcer.handleStreamError(nextError)
            await this.deps.restoreAfterError()
        }
    }
}

function isAbortError(error: Error): boolean {
    return error.name === "AbortError" || error.message.toLowerCase().includes("aborted")
}
