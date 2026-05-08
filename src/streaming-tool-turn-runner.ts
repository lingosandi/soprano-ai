import { coerceToolCallToDefinition } from "./command-response-utils"
import type { RunInvocationTarget } from "./command-response-utils"
import { StreamingToolParser } from "./streaming-tool-parser"
import { toError } from "./audio-helpers"
import type { AgentTurnRunner } from "./agent-turn-runner"
import type { LLMClient } from "./llm-client"
import type { SharedAgentRuntime } from "./shared-agent-runtime"
import type { ChatMessage, ToolCall } from "./types"

const EMPTY_RESPONSE_ERROR = "Empty response from API - will retry"

export interface StreamingTurnSpeechSink {
    sendSpeechChunk: (contextId: string, chunk: string) => Promise<void> | void
    flushSpeech?: (contextId: string) => Promise<void>
}

export interface StreamingToolTurnTransportDeps {
    runtime: SharedAgentRuntime
    llm: LLMClient
    getSystemPrompt: () => string
    getFallbackHistory: () => ChatMessage[]
    buildMessages?: (context: {
        text: string
        skipHistoryPush: boolean
        systemPrompt: string
        fallbackHistory: ChatMessage[]
    }) => ChatMessage[]
    temperature: number
    maxTokens: number
    requestExtraBody?: Record<string, unknown>
    allowedRunTargets?: readonly RunInvocationTarget[]
    speechBufferThreshold?: number
    sentenceEndPattern?: RegExp
    userMessageSaveReason?: string
    speechSink?: StreamingTurnSpeechSink
    onStreamStart?: () => void
    onFirstSpeechToken?: (options: { skipSpeech: boolean }) => void
    onPartialResponse?: (content: string) => void
    onToolCall?: (selectedToolCall: ToolCall) => void
    onToolCallError?: (error: Error, rawCommandText: string) => void
    handleToolCallParseError?: (
        error: Error,
        rawCommandText: string,
    ) => Promise<void> | void
    onStreamError?: (error: Error) => Promise<void> | void
    abortCurrentTurn?: () => void
    onLog?: (message: string) => void
    onError?: (error: Error) => void
}

export interface StreamingToolTurnRunnerDeps
    extends StreamingToolTurnTransportDeps {
    assistantMessageSaveReason?: string
    onFullResponse?: (content: string) => void
    transportRunner?: AgentTurnRunner<
        StreamingToolTurnRunOptions,
        StreamingToolTurnResult
    >
}

export interface StreamingToolTurnRunOptions {
    text: string
    contextId: string
    signal: AbortSignal
    skipHistoryPush?: boolean
    skipSpeech?: boolean
}

export interface StreamingToolTurnResult {
    aborted: boolean
    content: string
    pendingSelectedToolCall?: ToolCall
    speechSent: boolean
    continueWithoutTools?: boolean
    validationError?: string
}

export class StreamingToolTurnTransport
    implements AgentTurnRunner<StreamingToolTurnRunOptions, StreamingToolTurnResult>
{
    constructor(private deps: StreamingToolTurnTransportDeps) {}

    async run(
        options: StreamingToolTurnRunOptions,
    ): Promise<StreamingToolTurnResult> {
        const {
            text,
            contextId,
            signal,
            skipHistoryPush = false,
            skipSpeech = false,
        } = options

        if (!skipHistoryPush) {
            await this.deps.runtime.appendHistoryMessage({
                role: "user",
                content: text,
                reason:
                    this.deps.userMessageSaveReason ??
                    "Voice memory sync failed",
                fallbackHistory: this.deps.getFallbackHistory(),
            })
        }

        let speechBuffer = ""
        let speechSent = false
        let pendingSelectedToolCall: ToolCall | undefined
        const pendingValidationTasks: Promise<void>[] = []
        let validationError: string | undefined
        const toolDefinitions = this.deps.runtime.toolRegistry.getToolDefinitions()
        let turnResult: StreamingToolTurnResult | null = null
        const speechBufferThreshold = this.deps.speechBufferThreshold ?? 120
        const sentenceEndPattern = this.deps.sentenceEndPattern ?? /$^/

        const parser = new StreamingToolParser({
            allowedRunTargets: this.deps.allowedRunTargets,
            onSpeechToken: (speechToken) => {
                if (signal.aborted) {
                    return
                }

                this.deps.onPartialResponse?.(parser.getFullContent())

                if (!speechSent) {
                    speechSent = true
                    this.deps.onFirstSpeechToken?.({ skipSpeech })
                }

                if (skipSpeech || !this.deps.speechSink) {
                    return
                }

                speechBuffer += speechToken
                const endsWithPunctuation = sentenceEndPattern.test(
                    speechBuffer.trimEnd(),
                )

                if (
                    speechBuffer.length >= speechBufferThreshold ||
                    endsWithPunctuation
                ) {
                    const chunk = speechBuffer
                    speechBuffer = ""
                    void this.sendSpeechChunk(contextId, chunk)
                }
            },
            onToolCall: (selectedToolCall) => {
                if (signal.aborted) {
                    return
                }

                const coercedSelectedToolCall = coerceToolCallToDefinition(
                    selectedToolCall,
                    toolDefinitions,
                )
                if (!coercedSelectedToolCall) {
                    return
                }
                this.deps.onLog?.(
                    `Selected tool call detected: ${coercedSelectedToolCall.name}`,
                )
                this.deps.onToolCall?.(coercedSelectedToolCall)
                pendingSelectedToolCall = coercedSelectedToolCall
            },
            onToolCallError: (error, rawCommandText) => {
                if (signal.aborted) {
                    return
                }

                validationError = error.message
                this.deps.onToolCallError?.(error, rawCommandText)
                pendingValidationTasks.push(
                    Promise.resolve(
                        this.deps.handleToolCallParseError?.(error, rawCommandText),
                    )
                        .catch((parseError) => {
                            this.deps.onError?.(toError(parseError))
                        })
                        .then(() => {}),
                )
            },
            onLog: (message) => this.deps.onLog?.(message),
        })

        const systemPrompt = this.deps.getSystemPrompt()
        const fallbackHistory = this.deps.getFallbackHistory()
        const llmMessages = this.deps.buildMessages
            ? this.deps.buildMessages({
                text,
                skipHistoryPush,
                systemPrompt,
                fallbackHistory,
            })
            : this.deps.runtime.buildMessages({
                systemPrompt,
                fallbackHistory,
                coalesceAdjacentRoles: false,
            })

        this.deps.onLog?.(`Calling LLM (${llmMessages.length} messages)…`)

        await this.deps.llm.chat(
            {
                messages: llmMessages,
                temperature: this.deps.temperature,
                maxTokens: this.deps.maxTokens,
                extraBody: this.deps.requestExtraBody,
                stream: true,
                signal,
            },
            {
                onStreamStart: () => {
                    if (signal.aborted) {
                        return
                    }

                    this.deps.onStreamStart?.()
                },
                onToken: (token) => {
                    if (signal.aborted) {
                        return false
                    }

                    parser.push(token)

                    if (parser.sealed) {
                        return true
                    }

                    return false
                },
                onComplete: async (content, finishReason) => {
                    const acceptedContent = parser.getFullContent()

                    this.deps.onLog?.(
                        `onComplete: ${acceptedContent.length} chars, reason=${finishReason}, aborted=${signal.aborted}, selectedToolCall=${pendingSelectedToolCall ? 1 : 0}, speechSent=${speechSent}`,
                    )

                    if (signal.aborted) {
                        this.deps.onLog?.("onComplete: signal aborted — skipping")
                        turnResult = {
                            aborted: true,
                            content: acceptedContent,
                            ...(pendingSelectedToolCall ? { pendingSelectedToolCall } : {}),
                            speechSent,
                            continueWithoutTools: Boolean(validationError),
                            validationError,
                        }
                        return
                    }

                    parser.finish()

                    if (pendingValidationTasks.length > 0) {
                        await Promise.all(pendingValidationTasks)
                    }

                    if (!skipSpeech && speechBuffer.length > 0) {
                        await this.sendSpeechChunk(contextId, speechBuffer)
                        speechBuffer = ""
                    }

                    if (signal.aborted) {
                        turnResult = {
                            aborted: true,
                            content: acceptedContent,
                            ...(pendingSelectedToolCall ? { pendingSelectedToolCall } : {}),
                            speechSent,
                            continueWithoutTools: Boolean(validationError),
                            validationError,
                        }
                        return
                    }

                    if (speechSent && !skipSpeech) {
                        await this.deps.speechSink?.flushSpeech?.(contextId)
                    }

                    if (signal.aborted) {
                        turnResult = {
                            aborted: true,
                            content: acceptedContent,
                            ...(pendingSelectedToolCall ? { pendingSelectedToolCall } : {}),
                            speechSent,
                            continueWithoutTools: Boolean(validationError),
                            validationError,
                        }
                        return
                    }

                    turnResult = {
                        aborted: false,
                        content: acceptedContent,
                        ...(pendingSelectedToolCall ? { pendingSelectedToolCall } : {}),
                        speechSent,
                        continueWithoutTools: Boolean(validationError),
                        validationError,
                    }
                },
                onError: (error) => {
                    if (signal.aborted) {
                        return
                    }

                    this.deps.onLog?.(`LLM stream error: ${error.message}`)
                    this.deps.abortCurrentTurn?.()
                    void Promise.resolve(this.deps.onStreamError?.(error)).catch(
                        (streamError) => {
                            this.deps.onError?.(toError(streamError))
                        },
                    )
                },
                onLog: (message) => this.deps.onLog?.(message),
            },
        )

        return (
            turnResult ?? {
                aborted: signal.aborted,
                content: "",
                ...(pendingSelectedToolCall ? { pendingSelectedToolCall } : {}),
                speechSent,
                continueWithoutTools: Boolean(validationError),
                validationError,
            }
        )
    }

    private async sendSpeechChunk(
        contextId: string,
        chunk: string,
    ): Promise<void> {
        if (!chunk || !this.deps.speechSink) {
            return
        }

        try {
            await this.deps.speechSink.sendSpeechChunk(contextId, chunk)
        } catch (error) {
            this.deps.onError?.(toError(error))
        }
    }
}

export class StreamingToolTurnRunner
    implements AgentTurnRunner<StreamingToolTurnRunOptions, StreamingToolTurnResult>
{
    private transportRunner: AgentTurnRunner<
        StreamingToolTurnRunOptions,
        StreamingToolTurnResult
    >

    constructor(private deps: StreamingToolTurnRunnerDeps) {
        this.transportRunner =
            deps.transportRunner ?? new StreamingToolTurnTransport(deps)
    }

    async run(
        options: StreamingToolTurnRunOptions,
    ): Promise<StreamingToolTurnResult> {
        const result = await this.transportRunner.run(options)

        if (result.aborted) {
            return result
        }

        if (result.validationError) {
            return result
        }

        if (
            result.content.trim().length === 0
        ) {
            throw new Error(EMPTY_RESPONSE_ERROR)
        }

        try {
            await this.deps.runtime.appendHistoryMessage({
                role: "assistant",
                content: result.content,
                reason:
                    this.deps.assistantMessageSaveReason ??
                    "Voice memory save failed",
                immediateSync: true,
                fallbackHistory: this.deps.getFallbackHistory(),
            })
        } catch (memoryError) {
            this.deps.onError?.(toError(memoryError))
        }

        this.deps.onFullResponse?.(result.content)
        return result
    }
}