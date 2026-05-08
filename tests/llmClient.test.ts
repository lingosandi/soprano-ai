/**
 * Tests for packages/utils/llm-client.ts — LLMClient
 *
 * Uses a mock fetch to avoid real API calls.
 */
import { describe, expect, test } from "vitest"
import { LLMClient } from "../src/llm-client"
import type { LLMProviderConfig } from "../src/types"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeConfig(
    overrides?: Partial<LLMProviderConfig>
): LLMProviderConfig {
    return {
        apiKey: "test-key",
        baseUrl: "https://api.example.com",
        model: "test-model",
        ...overrides
    }
}

/** Build a mock fetch that returns a non-streaming JSON response. */
function mockFetchJson(data: any, status = 200): LLMProviderConfig["fetchImpl"] {
    return async (_url: string, _init?: any) => {
        return new Response(JSON.stringify(data), {
            status,
            headers: { "Content-Type": "application/json" }
        })
    }
}

/** Build a mock fetch that returns an SSE stream. */
function mockFetchSSE(
    chunks: string[],
    status = 200
): LLMProviderConfig["fetchImpl"] {
    return async (_url: string, _init?: any) => {
        const encoder = new TextEncoder()
        const stream = new ReadableStream({
            start(controller) {
                for (const chunk of chunks) {
                    controller.enqueue(encoder.encode(chunk))
                }
                controller.close()
            }
        })
        return new Response(stream, {
            status,
            headers: { "Content-Type": "text/event-stream" }
        })
    }
}

/** Create SSE `data:` lines from token strings.  Ends with [DONE]. */
function sseFromTokens(tokens: string[]): string[] {
    const lines: string[] = []
    for (const token of tokens) {
        const obj = { choices: [{ delta: { content: token } }] }
        lines.push(`data: ${JSON.stringify(obj)}\n\n`)
    }
    lines.push("data: [DONE]\n\n")
    return lines
}

// ---------------------------------------------------------------------------
// Non-streaming
// ---------------------------------------------------------------------------

describe("LLMClient — non-streaming", () => {
    test("returns parsed content from JSON response", async () => {
        const fetchImpl = mockFetchJson({
            choices: [
                {
                    message: { content: "Hello from LLM" },
                    finish_reason: "stop"
                }
            ],
            usage: { prompt_tokens: 10, completion_tokens: 5 }
        })

        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat({
            messages: [{ role: "user", content: "Hi" }],
            stream: false
        })

        expect(result.content).toBe("Hello from LLM")
        expect(result.finishReason).toBe("stop")
        expect(result.usage?.promptTokens).toBe(10)
        expect(result.usage?.completionTokens).toBe(5)
    })

    test("defaults finish_reason to 'stop' when missing", async () => {
        const fetchImpl = mockFetchJson({
            choices: [{ message: { content: "Hi" } }]
        })
        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat({
            messages: [{ role: "user", content: "Hi" }],
            stream: false
        })
        expect(result.finishReason).toBe("stop")
    })

    test("returns empty content when no choices", async () => {
        const fetchImpl = mockFetchJson({ choices: [] })
        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat({
            messages: [{ role: "user", content: "Hi" }],
            stream: false
        })
        expect(result.content).toBe("")
    })
})

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe("LLMClient — streaming", () => {
    test("fires onToken for each chunk and onComplete at end", async () => {
        const tokens = ["Hello", " ", "world"]
        const fetchImpl = mockFetchSSE(sseFromTokens(tokens))

        const receivedTokens: string[] = []
        let completeCalled = false
        let completeContent = ""

        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }], stream: true },
            {
                onToken: (t) => {
                    receivedTokens.push(t)
                },
                onComplete: (content) => {
                    completeCalled = true
                    completeContent = content
                }
            }
        )

        expect(receivedTokens).toEqual(["Hello", " ", "world"])
        expect(completeCalled).toBe(true)
        expect(completeContent).toBe("Hello world")
        expect(result.content).toBe("Hello world")
    })

    test("captures finish_reason from stream", async () => {
        const obj = {
            choices: [{ delta: { content: "Hi" }, finish_reason: "length" }]
        }
        const fetchImpl = mockFetchSSE([
            `data: ${JSON.stringify(obj)}\n\n`,
            "data: [DONE]\n\n"
        ])

        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }] },
            {}
        )

        expect(result.finishReason).toBe("length")
    })

    test("skips malformed SSE chunks without crashing", async () => {
        const lines = [
            "data: NOT_JSON\n\n",
            `data: ${JSON.stringify({ choices: [{ delta: { content: "ok" } }] })}\n\n`,
            "data: [DONE]\n\n"
        ]
        const fetchImpl = mockFetchSSE(lines)
        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }] },
            {}
        )
        expect(result.content).toBe("ok")
    })

    test("ignores non-data lines", async () => {
        const lines = [
            ": comment\n\n",
            "event: ping\n\n",
            `data: ${JSON.stringify({ choices: [{ delta: { content: "yes" } }] })}\n\n`,
            "data: [DONE]\n\n"
        ]
        const fetchImpl = mockFetchSSE(lines)
        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }] },
            {}
        )
        expect(result.content).toBe("yes")
    })

    test("fires onStreamStart once on first SSE chunk", async () => {
        const tokens = ["Hello", " ", "world"]
        const fetchImpl = mockFetchSSE(sseFromTokens(tokens))

        let streamStartCount = 0
        const client = new LLMClient(makeConfig({ fetchImpl }))
        await client.chat(
            { messages: [{ role: "user", content: "Hi" }], stream: true },
            {
                onStreamStart: () => { streamStartCount++ },
            }
        )

        expect(streamStartCount).toBe(1)
    })

    test("fires onStreamStart before first onToken", async () => {
        const order: string[] = []
        const fetchImpl = mockFetchSSE(sseFromTokens(["A", "B"]))
        const client = new LLMClient(makeConfig({ fetchImpl }))
        await client.chat(
            { messages: [{ role: "user", content: "Hi" }], stream: true },
            {
                onStreamStart: () => { order.push("streamStart") },
                onToken: () => { order.push("token") },
            }
        )

        expect(order[0]).toBe("streamStart")
        expect(order.filter(e => e === "streamStart").length).toBe(1)
    })

    test("reasoning-only deltas do not fire onToken but stream still completes", async () => {
        // Simulate Qwen 3 thinking mode: reasoning_content tokens
        // arrive in delta but delta.content is absent/empty.
        const lines = [
            `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: "Let me think..." } }] })}\n\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { reasoning_content: "about this" } }] })}\n\n`,
            `data: ${JSON.stringify({ choices: [{ delta: { content: "Answer" } }] })}\n\n`,
            "data: [DONE]\n\n",
        ]
        const fetchImpl = mockFetchSSE(lines)

        const receivedTokens: string[] = []
        let streamStarted = false

        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }], stream: true },
            {
                onStreamStart: () => { streamStarted = true },
                onToken: (t) => {
                    receivedTokens.push(t)
                },
            }
        )

        // onStreamStart fires on the first chunk (even though it's a reasoning token)
        expect(streamStarted).toBe(true)
        // Only the content token is forwarded via onToken
        expect(receivedTokens).toEqual(["Answer"])
        // fullContent only includes delta.content
        expect(result.content).toBe("Answer")
    })
})

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

describe("LLMClient — request construction", () => {
    test("sends correct URL, headers, and body", async () => {
        let capturedUrl: string | undefined
        let capturedInit: any

        const fetchImpl: LLMProviderConfig["fetchImpl"] = async (
            url: string,
            init?: any
        ) => {
            capturedUrl = url
            capturedInit = init
            return new Response(
                JSON.stringify({
                    choices: [{ message: { content: "" }, finish_reason: "stop" }]
                }),
                { status: 200 }
            )
        }

        const client = new LLMClient(
            makeConfig({
                fetchImpl,
                baseUrl: "https://api.test.com",
                apiKey: "my-key",
                model: "my-model",
                extraHeaders: { "X-Custom": "val" },
                extraBody: { provider_flag: "from-config", enable_thinking: true }
            })
        )

        await client.chat({
            messages: [{ role: "user", content: "test" }],
            stream: false,
            temperature: 0.5,
            maxTokens: 1000,
            extraBody: { enable_thinking: false }
        })

        expect(capturedUrl).toBe("https://api.test.com/chat/completions")
        const headers = capturedInit.headers
        expect(headers["Authorization"]).toBe("Bearer my-key")
        expect(headers["Content-Type"]).toBe("application/json")
        expect(headers["X-Custom"]).toBe("val")

        const body = JSON.parse(capturedInit.body)
        expect(body.model).toBe("my-model")
        expect(body.temperature).toBe(0.5)
        expect(body.max_tokens).toBe(1000)
        expect(body.stream).toBe(false)
        expect(body.provider_flag).toBe("from-config")
        expect(body.enable_thinking).toBe(false)
    })

    test("omits temperature for kimi-k2.5 requests", async () => {
        let capturedInit: any

        const fetchImpl: LLMProviderConfig["fetchImpl"] = async (
            _url: string,
            init?: any,
        ) => {
            capturedInit = init
            return new Response(
                JSON.stringify({
                    choices: [{ message: { content: "" }, finish_reason: "stop" }],
                }),
                { status: 200 },
            )
        }

        const client = new LLMClient(
            makeConfig({
                fetchImpl,
                baseUrl: "https://api.moonshot.cn/v1",
                model: "kimi-k2.5",
            }),
        )

        await client.chat({
            messages: [{ role: "user", content: "test" }],
            stream: false,
            temperature: 1,
            maxTokens: 1000,
        })

        const body = JSON.parse(capturedInit.body)
        expect(body.model).toBe("kimi-k2.5")
        expect(body.temperature).toBeUndefined()
        expect(body.max_tokens).toBe(1000)
        expect(body.stream).toBe(false)
    })
})

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe("LLMClient — error handling", () => {
    test("throws on non-200 response (non-retryable)", async () => {
        const fetchImpl = mockFetchJson(
            { error: "Unauthorized" },
            401
        )
        const client = new LLMClient(makeConfig({ fetchImpl }))

        await expect(
            client.chat({
                messages: [{ role: "user", content: "Hi" }],
                stream: false
            })
        ).rejects.toThrow("API error 401")
    })

    test("throws on 400 response", async () => {
        const fetchImpl = mockFetchJson({ error: "Bad Request" }, 400)
        const client = new LLMClient(makeConfig({ fetchImpl }))

        await expect(
            client.chat({
                messages: [{ role: "user", content: "Hi" }],
                stream: false
            })
        ).rejects.toThrow("API error 400")
    })

    test("retries on network error (fetch failed)", async () => {
        let callCount = 0
        const fetchImpl: LLMProviderConfig["fetchImpl"] = async () => {
            callCount++
            if (callCount < 3) {
                const err = new Error("fetch failed")
                throw err
            }
            return new Response(
                JSON.stringify({
                    choices: [
                        {
                            message: { content: "recovered" },
                            finish_reason: "stop"
                        }
                    ]
                }),
                { status: 200 }
            )
        }

        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat({
            messages: [{ role: "user", content: "Hi" }],
            stream: false
        })

        expect(result.content).toBe("recovered")
        expect(callCount).toBe(3)
    }, 15000)

    test("throws after max retries exhausted", async () => {
        let callCount = 0
        const fetchImpl: LLMProviderConfig["fetchImpl"] = async () => {
            callCount++
            const err = new Error("fetch failed")
            throw err
        }

        const client = new LLMClient(makeConfig({ fetchImpl }))

        await expect(
            client.chat({
                messages: [{ role: "user", content: "Hi" }],
                stream: false
            })
        ).rejects.toThrow("fetch failed")

        expect(callCount).toBe(3) // 3 attempts
    }, 15000)
})

// ---------------------------------------------------------------------------
// Config management
// ---------------------------------------------------------------------------

describe("LLMClient — config", () => {
    test("setConfig swaps the provider", () => {
        const client = new LLMClient(makeConfig())
        const newConfig = makeConfig({ model: "new-model" })
        client.setConfig(newConfig)
        expect(client.getConfig().model).toBe("new-model")
    })

    test("getConfig returns current config", () => {
        const config = makeConfig({ apiKey: "my-key" })
        const client = new LLMClient(config)
        expect(client.getConfig().apiKey).toBe("my-key")
    })
})

// ---------------------------------------------------------------------------
// Null body handling
// ---------------------------------------------------------------------------

describe("LLMClient — edge cases", () => {
    test("throws when response body is null (streaming)", async () => {
        const fetchImpl: LLMProviderConfig["fetchImpl"] = async () => {
            return new Response(null, { status: 200 })
        }
        const client = new LLMClient(makeConfig({ fetchImpl }))
        await expect(
            client.chat({
                messages: [{ role: "user", content: "Hi" }],
                stream: true
            })
        ).rejects.toThrow()
    })

    test("handles empty stream (no tokens before [DONE])", async () => {
        const lines = ["data: [DONE]\n\n"]
        const fetchImpl = mockFetchSSE(lines)
        const client = new LLMClient(makeConfig({ fetchImpl }))
        const result = await client.chat(
            { messages: [{ role: "user", content: "Hi" }] },
            {}
        )
        expect(result.content).toBe("")
    })

    test("awaits async onComplete before resolving chat()", async () => {
        const tokens = ["Hello"]
        const fetchImpl = mockFetchSSE(sseFromTokens(tokens))

        const order: string[] = []

        const client = new LLMClient(makeConfig({ fetchImpl }))
        await client.chat(
            { messages: [{ role: "user", content: "Hi" }], stream: true },
            {
                onComplete: async () => {
                    // Simulate async work (e.g. TTS flush, tool execution)
                    await new Promise((r) => setTimeout(r, 50))
                    order.push("onComplete-done")
                },
            }
        )
        order.push("chat-resolved")

        // onComplete must fully resolve BEFORE chat() resolves
        expect(order).toEqual(["onComplete-done", "chat-resolved"])
    })

    test("onComplete error propagates to chat() caller", async () => {
        const tokens = ["Hi"]
        const fetchImpl = mockFetchSSE(sseFromTokens(tokens))

        const client = new LLMClient(makeConfig({ fetchImpl }))

        await expect(
            client.chat(
                { messages: [{ role: "user", content: "Hi" }], stream: true },
                {
                    onComplete: async () => {
                        throw new Error("TTS flush failed")
                    },
                }
            )
        ).rejects.toThrow("TTS flush failed")
    })
})
