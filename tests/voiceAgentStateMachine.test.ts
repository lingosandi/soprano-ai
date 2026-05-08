/**
 * VoiceAgentService state-machine and API tests.
 *
 * This suite focuses on the foreground voice behavior after the passive
 * background-analysis feature stack was removed.
 */

import { afterEach, beforeEach, describe, expect, test } from "vitest"
import {
    ASR_LANGUAGE_HINTS,
    ASR_MAX_SENTENCE_SILENCE_MS,
    ASR_SPEECH_NOISE_THRESHOLD,
} from "../src/config"
import type { ToolRegistry } from "../src/tool-registry"
import { VoiceAgentService } from "../src"
import type {
    ASRCallbacks,
    ASRStartOptions,
    BackgroundToolResult,
    IAudioPlayer,
    IASRService,
} from "../src"
import type { ToolCall, ToolDefinition } from "../src/types"
import { createTestToolRegistry } from "./helpers/createTestToolRegistry"

const toolSegment = (payload: string) => ["---tool---", payload].join("\n")
const TEST_API_KEYS = {
    qwenApiKey: "test-qwen-key",
    cartesiaApiKey: "test-cartesia-key",
}

const TEST_TOOL_DEFINITION: ToolDefinition = {
    name: "LookupItem",
    description: "Look up a test item.",
    parameters: {
        itemId: {
            type: "string",
            required: true,
            description: "Item id to look up.",
        },
    },
}

class MockWebSocket {
    static OPEN = 1
    static CLOSED = 3

    url: string
    readyState = MockWebSocket.OPEN
    onopen: ((ev: any) => void) | null = null
    onclose: ((ev: any) => void) | null = null
    onerror: ((ev: any) => void) | null = null
    onmessage: ((ev: any) => void) | null = null
    binaryType = "arraybuffer"
    sentMessages: any[] = []

    constructor(url: string) {
        this.url = url
        queueMicrotask(() => this.onopen?.({}))
    }

    send(data: any) {
        this.sentMessages.push(data)
        if (typeof data !== "string") {
            return
        }

        try {
            const msg = JSON.parse(data)
            if (msg.transcript !== undefined && msg.context_id) {
                const contextId = msg.context_id
                queueMicrotask(() => {
                    this.onmessage?.({
                        data: JSON.stringify({
                            type: "chunk",
                            context_id: contextId,
                            data: btoa(String.fromCharCode(...new Array(64).fill(0))),
                            step_time: 0.05,
                        }),
                    })
                })
                if (msg.continue === false) {
                    queueMicrotask(() => {
                        this.onmessage?.({
                            data: JSON.stringify({
                                type: "done",
                                context_id: contextId,
                            }),
                        })
                    })
                }
            }
        } catch {
            // Ignore non-JSON websocket messages in tests.
        }
    }

    close() {
        this.readyState = MockWebSocket.CLOSED
        this.onclose?.({})
    }
}

class MockAudioPlayer implements IAudioPlayer {
    initialized = false
    enqueuedChunks: Float32Array[] = []
    stopped = false
    _isPlaying = false
    audioLevel = 0.5
    private endResolvers: Array<() => void> = []

    async init() {
        this.initialized = true
        this.stopped = false
    }

    enqueue(samples: Float32Array) {
        this.enqueuedChunks.push(samples)
        this._isPlaying = true
    }

    stop() {
        this.stopped = true
        this._isPlaying = false
        for (const resolve of this.endResolvers) {
            resolve()
        }
        this.endResolvers = []
    }

    getAudioLevel() {
        return this.audioLevel
    }

    get isPlaying() {
        return this._isPlaying
    }

    async waitForEnd() {
        if (!this._isPlaying) return
        return new Promise<void>((resolve) => {
            this.endResolvers.push(resolve)
            setTimeout(() => {
                this._isPlaying = false
                resolve()
            }, 10)
        })
    }
}

class MockASRService implements IASRService {
    callbacks: ASRCallbacks = {}
    isRunning = false
    startCallCount = 0
    stopCallCount = 0
    destroyCallCount = 0
    startOptions: ASRStartOptions[] = []
    _isMicMuted = false
    micMutedCallArgs: boolean[] = []
    private failsRemaining = 0

    on(cb: ASRCallbacks) {
        this.callbacks = { ...this.callbacks, ...cb }
    }

    async start(_apiKey: string, opts?: ASRStartOptions) {
        this.startCallCount++
        if (this.failsRemaining > 0) {
            this.failsRemaining--
            throw new Error("ASR connection failed")
        }
        if (opts) {
            this.startOptions.push({ ...opts })
        }
        this.isRunning = true
    }

    async stop() {
        this.stopCallCount++
        this.isRunning = false
    }

    destroy() {
        this.destroyCallCount++
        this.isRunning = false
    }

    setMicMuted(muted: boolean) {
        this.micMutedCallArgs.push(muted)
        this._isMicMuted = muted
    }

    get isMicMuted() {
        return this._isMicMuted
    }

    failNTimes(count: number) {
        this.failsRemaining = count
    }

    emitInterim(text: string) {
        this.callbacks.onInterim?.(text)
    }

    emitFinal(text: string) {
        this.callbacks.onFinal?.(text)
    }

    emitError(err: Error) {
        this.callbacks.onError?.(err)
    }

    emitFinished() {
        this.callbacks.onFinished?.()
    }
}

function makeLLMFetch(
    responseText = "Hi there!",
    opts?: {
        shouldFail?: boolean
        delayMs?: number
        commandSuffix?: string
        toolCallOnce?: boolean
    },
) {
    let toolCallEmitted = false
    const toolCallOnce = opts?.toolCallOnce !== false
    const calls: Array<{ url: string; init?: any }> = []

    function assertAllowedLLMTestUrl(url: string) {
        const parsed = new URL(url)
        const isProviderRequest = (
            (parsed.hostname === "api.minimaxi.com" || parsed.hostname === "dashscope.aliyuncs.com")
            && parsed.pathname.endsWith("/chat/completions")
        )

        if (!isProviderRequest) {
            throw new Error(`Unexpected LLM fetch URL in VoiceAgentStateMachine test: ${url}`)
        }
    }

    const fetch = async (url: string, init?: any): Promise<Response> => {
        assertAllowedLLMTestUrl(url)
        calls.push({ url, init })
        if (opts?.shouldFail) {
            return new Response('{"error":"fail"}', { status: 500 })
        }
        if (opts?.delayMs) {
            await new Promise((resolve) => setTimeout(resolve, opts.delayMs))
        }

        const includeCommandSuffix = Boolean(opts?.commandSuffix) && (!toolCallOnce || !toolCallEmitted)
        if (includeCommandSuffix) {
            toolCallEmitted = true
        }

        const content = includeCommandSuffix
            ? `${responseText}${opts!.commandSuffix}`
            : responseText || "Sure."

        const sseLines = [
            ...content.split("").map((character) =>
                `data: ${JSON.stringify({ choices: [{ delta: { content: character } }] })}`,
            ),
            `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}`,
            "data: [DONE]",
        ]

        const stream = new ReadableStream({
            start(controller) {
                controller.enqueue(new TextEncoder().encode(sseLines.join("\n") + "\n"))
                controller.close()
            },
        })

        return new Response(stream, {
            status: 200,
            headers: { "Content-Type": "text/event-stream" },
        })
    }

    return { fetch, calls }
}

interface Deps {
    player: MockAudioPlayer
    asr: MockASRService
    states: string[]
    logs: string[]
    errors: Error[]
    transcripts: string[]
    partials: string[]
    fullResponses: string[]
    emittedCalls: ToolCall[]
    bgResults: BackgroundToolResult[]
    userMessages: string[]
    audioLevels: number[]
}

function makeAgent(
    fetchFn: (url: string, init?: any) => Promise<Response>,
    opts?: {
        toolRegistry?: ToolRegistry
        backgroundTaskPoller?: () => Promise<BackgroundToolResult[]>
        memory?: any
        asrOptions?: Partial<ASRStartOptions>
    },
): { svc: VoiceAgentService; deps: Deps } {
    const player = new MockAudioPlayer()
    const asr = new MockASRService()
    const states: string[] = []
    const logs: string[] = []
    const errors: Error[] = []
    const transcripts: string[] = []
    const partials: string[] = []
    const fullResponses: string[] = []
    const emittedCalls: ToolCall[] = []
    const bgResults: BackgroundToolResult[] = []
    const userMessages: string[] = []
    const audioLevels: number[] = []

    const svc = new VoiceAgentService({
        player,
        asr,
        apiKeys: TEST_API_KEYS,
        fetchImpl: fetchFn,
        toolRegistry: opts?.toolRegistry,
        backgroundTaskPoller: opts?.backgroundTaskPoller,
        memory: opts?.memory,
        asrOptions: opts?.asrOptions,
    })

    svc.on({
        onStateChange: (state) => states.push(state),
        onLog: (message) => logs.push(message),
        onError: (error) => errors.push(error),
        onUserTranscript: (text) => transcripts.push(text),
        onPartialResponse: (text) => partials.push(text),
        onFullResponse: (text) => fullResponses.push(text),
        onToolCall: (selectedToolCall) => emittedCalls.push(selectedToolCall),
        onBackgroundTaskComplete: (result) => bgResults.push(result),
        onUserMessage: (message) => userMessages.push(message),
        onAudioLevel: (level) => audioLevels.push(level),
    })

    return {
        svc,
        deps: {
            player,
            asr,
            states,
            logs,
            errors,
            transcripts,
            partials,
            fullResponses,
            emittedCalls,
            bgResults,
            userMessages,
            audioLevels,
        },
    }
}

function settle(ms = 50) {
    return new Promise<void>((resolve) => setTimeout(resolve, ms))
}

async function initToListening(svc: VoiceAgentService) {
    await svc.init()
    await settle(100)
    svc.interrupt()
    await svc.startListening()
    await settle()
}

let savedWebSocket: any

beforeEach(() => {
    savedWebSocket = (globalThis as any).WebSocket
    ;(globalThis as any).WebSocket = MockWebSocket
})

afterEach(() => {
    ;(globalThis as any).WebSocket = savedWebSocket
})

describe("VoiceAgentService — mic mute API", () => {
    test("isMicMuted defaults to false and reflects explicit service mute state", () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        expect(svc.isMicMuted).toBe(false)
        svc.setMicMuted(true)
        expect(svc.isMicMuted).toBe(true)

        deps.asr._isMicMuted = true
        expect(svc.isMicMuted).toBe(true)
        svc.destroy()
    })

    test("setMicMuted delegates to asr.setMicMuted", () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        svc.setMicMuted(true)
        svc.setMicMuted(false)

        expect(deps.asr.micMutedCallArgs).toEqual([true, false])
        expect(deps.asr._isMicMuted).toBe(false)
        svc.destroy()
    })

    test("setMicMuted is safe on ASR service without setMicMuted support", () => {
        const { fetch } = makeLLMFetch("Hi")
        const player = new MockAudioPlayer()
        const asr: IASRService = {
            on() {},
            async start() {},
            async stop() {},
            destroy() {},
        }

        const svc = new VoiceAgentService({ player, asr, apiKeys: TEST_API_KEYS, fetchImpl: fetch })

        expect(() => svc.setMicMuted(true)).not.toThrow()
        expect(svc.isMicMuted).toBe(true)
        svc.destroy()
    })
})

describe("VoiceAgentService — muted transcript guard", () => {
    test("ASR final transcript is ignored when microphone is muted", async () => {
        const { fetch, calls } = makeLLMFetch("Response")
        const { svc, deps } = makeAgent(fetch)

        await initToListening(svc)
        const callsBefore = calls.length

        svc.setMicMuted(true)
        deps.asr.emitFinal("This should be ignored")
        await settle(100)

        expect(calls.length).toBe(callsBefore)
        expect(deps.logs.some((entry) => entry.includes("Ignoring transcript while muted"))).toBe(true)
        svc.destroy()
    })

    test("unmuting after mute allows transcripts through again", async () => {
        const { fetch, calls } = makeLLMFetch("Response")
        const { svc, deps } = makeAgent(fetch)

        await initToListening(svc)
        svc.setMicMuted(true)
        deps.asr.emitFinal("ignored")
        svc.setMicMuted(false)
        await settle(50)

        const callsBefore = calls.length
        deps.asr.emitFinal("processed")
        await settle(200)

        expect(calls.length).toBeGreaterThan(callsBefore)
        expect(deps.transcripts).toContain("processed")
        svc.destroy()
    })
})

describe("VoiceAgentService — asrOptions passthrough", () => {
    test("custom maxSilenceMs is passed to asr.start", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch, {
            asrOptions: { maxSilenceMs: 2500 },
        })

        await svc.startListening()

        expect(deps.asr.startOptions.at(-1)?.maxSilenceMs).toBe(2500)
        svc.destroy()
    })

    test("default ASR options come from shared voice constants", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        await svc.startListening()

        expect(deps.asr.startOptions.at(-1)).toEqual({
            maxSilenceMs: ASR_MAX_SENTENCE_SILENCE_MS,
            speechNoiseThreshold: ASR_SPEECH_NOISE_THRESHOLD,
            languageHints: ASR_LANGUAGE_HINTS,
        })
        svc.destroy()
    })
})

describe("VoiceAgentService — startListening guard", () => {
    test("calling startListening twice does not start ASR twice", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        await svc.startListening()
        await svc.startListening()

        expect(deps.asr.startCallCount).toBe(1)
        svc.destroy()
    })

    test("startListening stays in connecting until ASR start completes", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const player = new MockAudioPlayer()
        let startResolver!: () => void
        const startGate = new Promise<void>((resolve) => {
            startResolver = resolve
        })

        class DelayedStartASR extends MockASRService {
            override async start(apiKey: string, opts?: ASRStartOptions) {
                await startGate
                return super.start(apiKey, opts)
            }
        }

        const asr = new DelayedStartASR()
        const svc = new VoiceAgentService({ player, asr, apiKeys: TEST_API_KEYS, fetchImpl: fetch })

        const startPromise = svc.startListening()
        await settle(10)
        expect(svc.state).toBe("connecting")

        startResolver()
        await startPromise
        expect(svc.state).toBe("listening")
        svc.destroy()
    })
})

describe("VoiceAgentService — system prompt tool instructions", () => {
    test("LLM request includes tool instructions in the system message when a tool registry is provided", async () => {
        const { fetch, calls } = makeLLMFetch("Sure.")
        const toolRegistry = createTestToolRegistry([TEST_TOOL_DEFINITION])
        const { svc } = makeAgent(fetch, { toolRegistry })

        await svc.sendTextOnly("check tools")
        await settle(100)

        const requestBody = JSON.parse(String(calls.at(-1)?.init?.body ?? "{}"))
        const systemMessage = requestBody.messages.find((message: any) => message.role === "system")

        expect(systemMessage?.content).toContain("## Available Tools")
        expect(systemMessage?.content).toContain("LookupItem <itemId>")
        svc.destroy()
    })
})

describe("VoiceAgentService — sendTextOnly pipeline", () => {
    test("sendTextOnly produces thinking, speaking, and idle states", async () => {
        const { fetch } = makeLLMFetch("Response")
        const { svc, deps } = makeAgent(fetch)

        await svc.sendTextOnly("Hello")
        await settle(100)

        expect(deps.userMessages).toContain("Hello")
        expect(deps.states).toContain("thinking")
        expect(deps.states).toContain("speaking")
        expect(svc.state).toBe("idle")
        svc.destroy()
    })

    test("tool-call responses are emitted through onToolCall", async () => {
        const { fetch } = makeLLMFetch("Let me check.\n", {
                commandSuffix: toolSegment("LookupItem abc123"),
        })
        const toolRegistry = createTestToolRegistry(
            [TEST_TOOL_DEFINITION],
            async () => ({ success: true, output: "done" }),
        )
        const { svc, deps } = makeAgent(fetch, { toolRegistry })

        await svc.sendTextOnly("status")
        await settle(100)

        expect(deps.emittedCalls.length).toBeGreaterThan(0)
        expect(deps.emittedCalls.at(-1)).toEqual({
            kind: "tool",
            name: "LookupItem",
            parameters: { itemId: "abc123" },
            rawInvocation: "LookupItem abc123",
        })
        svc.destroy()
    })
})

describe("VoiceAgentService — continuousMode=false", () => {
    test("startListening(false) disables auto mic restart after a response", async () => {
        const { fetch } = makeLLMFetch("Hi there")
        const { svc, deps } = makeAgent(fetch)

        await svc.init()
        await settle(100)
        svc.interrupt()
        await svc.startListening(false)
        await settle()
        const startsBeforeResponse = deps.asr.startCallCount
        deps.asr.emitFinal("one turn only")
        await settle(250)

        expect(deps.asr.startCallCount).toBe(startsBeforeResponse)
        expect(svc.state).toBe("idle")
        svc.destroy()
    })
})

describe("VoiceAgentService — getAudioLevel", () => {
    test("getAudioLevel returns the player's current level", () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        deps.player.audioLevel = 0.73
        expect(svc.getAudioLevel()).toBe(0.73)
        svc.destroy()
    })
})

describe("VoiceAgentService — destroy", () => {
    test("destroy transitions the service to idle and destroys ASR", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        await svc.startListening()
        svc.destroy()

        expect(svc.state).toBe("idle")
        expect(deps.asr.destroyCallCount).toBe(1)
    })
})

describe("VoiceAgentService — ASR error propagation", () => {
    test("ASR error after init fires onError", async () => {
        const { fetch } = makeLLMFetch("Hi")
        const { svc, deps } = makeAgent(fetch)

        await initToListening(svc)
        deps.asr.emitError(new Error("microphone lost"))
        await settle(50)

        expect(deps.errors.some((error) => error.message.includes("microphone lost"))).toBe(true)
        svc.destroy()
    })
})
