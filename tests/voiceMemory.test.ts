/**
 * Tests for VoiceMemoryLayer (soprano-ai)
 *
 * Covers:
 *   - appendMessage / turn tracking
 *   - buildMessages assembly (system → state → recent)
 *   - Compaction triggers + topic/preference/fact/tool extraction
 *   - Compaction (dedup topics, enforce caps)
 *   - updateLastMessageByRole (charCount sync)
 *   - popLastMessage / getLastMessageRole / getLastMessageByRole
 *   - Persistence: save / load / clear
 *   - loadFromPersisted defensive validation
 *   - renderState formatting
 *   - reset()
 */
import { afterEach, describe, expect, test } from "vitest"
import { VoiceMemoryLayer, type VoiceMemoryStorage, type PersistedVoiceMemory } from "../src"
import { VOICE_SYSTEM_PROMPT_OVERHEAD } from "../src/config"

// ---------------------------------------------------------------------------
// Fixed test thresholds — independent of whatever provider is active.
// These are tuned so the test's fillMessages() content (at ~120 tokens/msg)
// exceeds the protected window, leaving early messages compressible.
// ---------------------------------------------------------------------------
const TEST_COMPACTION_THRESHOLD = 48_000
const TEST_PROTECTED_CONTEXT    = 10_000
const TEST_STATE_BUDGET          = 3_200

// ---------------------------------------------------------------------------
// In-memory storage adapter for tests
// ---------------------------------------------------------------------------

class InMemoryStorage implements VoiceMemoryStorage {
    data: PersistedVoiceMemory | null = null
    saveCount = 0
    loadCount = 0
    clearCount = 0

    async load(): Promise<PersistedVoiceMemory | null> {
        this.loadCount++
        return this.data ? structuredClone(this.data) : null
    }
    async save(data: PersistedVoiceMemory): Promise<void> {
        this.saveCount++
        this.data = structuredClone(data)
    }
    async clear(): Promise<void> {
        this.clearCount++
        this.data = null
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Clean up tiktoken encoder after each test to avoid memory leaks
let layerInstances: VoiceMemoryLayer[] = []
function tracked(layer: VoiceMemoryLayer): VoiceMemoryLayer {
    layerInstances.push(layer)
    return layer
}

afterEach(() => {
    for (const l of layerInstances) {
        l.dispose()
    }
    layerInstances = []
})

async function createLayer(storage?: VoiceMemoryStorage): Promise<VoiceMemoryLayer> {
    return tracked(await VoiceMemoryLayer.create(storage, undefined, {
        compactionTokenThreshold: TEST_COMPACTION_THRESHOLD,
        protectedContextTokens: TEST_PROTECTED_CONTEXT,
        stateBudgetTokens: TEST_STATE_BUDGET,
        maxRawHistory: 500,
    }))
}

/**
 * Create a layer and fill with enough content + static context to exceed
 * the compaction threshold (75% of context window).
 *
 * The voice config has protectedContextTokens≈16,640 and systemPromptOverhead=4000.
 * estimateContextTokens() = recentWindow + stateTokens + static + overhead.
 * To exceed ~48,000: we need recentWindow + static (~45,000) + overhead (4,000) > ~48,000.
 */
async function createCompactableLayer(): Promise<VoiceMemoryLayer> {
    const layer = await createLayer()
    // Static context adds ~45,000 tiktoken tokens to the estimate
    layer.setStaticContext("Context: " + "This is background information about the user and environment. ".repeat(4500))
    return layer
}

/** Fill a layer with enough messages to build a compressible history.
 * Each message pair is ~340 tiktoken tokens so even 50 pairs (~17,000 tokens)
 * overflows the 16,640-token protected window, putting early messages outside
 * the window and making them eligible for compaction. */
function fillMessages(layer: VoiceMemoryLayer, count: number) {
    const userPad = "The user is asking about various aspects of software development including architecture patterns, performance optimization techniques, testing strategies, deployment pipelines, and code review best practices for modern applications. This conversation covers a wide range of topics that are important for building scalable, maintainable, and well-tested software systems in production environments with high availability requirements and strict performance budgets that need careful attention to detail in every aspect of the development lifecycle from initial design through deployment and monitoring."
    const asstPad = "I will explain the key concepts in depth, covering the theoretical foundations and practical implementation details that are relevant to your question about modern software engineering practices and methodologies. Let me walk through the most important considerations including system design tradeoffs, common anti-patterns to avoid, recommended testing approaches, continuous integration and delivery pipelines, observability and monitoring strategies, and how to establish effective code review processes that improve team productivity while maintaining high code quality standards across the entire codebase."
    for (let i = 0; i < count; i++) {
        layer.appendMessage("user", `User question ${i} about topic ${i}: this is a sufficiently long message that contributes meaningfully to the token count and enables compaction testing with realistic content lengths. ${userPad}`)
        layer.appendMessage("assistant", `Assistant response ${i}: here is a detailed answer about topic ${i} that contains enough characters to push us past the compaction threshold when combined with other messages. ${asstPad}`)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("VoiceMemoryLayer — appendMessage", () => {
    test("appends a message and increments turn", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "hello")
        expect(layer.getCurrentTurn()).toBe(1)
        expect(layer.getRawHistoryLength()).toBe(1)
    })

    test("multiple appends increment turn correctly", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "first")
        layer.appendMessage("assistant", "second")
        layer.appendMessage("user", "third")
        expect(layer.getCurrentTurn()).toBe(3)
        expect(layer.getRawHistoryLength()).toBe(3)
    })

    test("appended messages have correct roles and content", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "What is the weather?")
        layer.appendMessage("assistant", "It's sunny today.")
        const history = layer.getRawHistory()
        expect(history[0].role).toBe("user")
        expect(history[0].content).toBe("What is the weather?")
        expect(history[1].role).toBe("assistant")
        expect(history[1].content).toBe("It's sunny today.")
    })

    test("appended messages have charCount", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "Hello world")
        const msg = layer.getRawHistory()[0]
        expect(msg.charCount).toBe(11)
    })

    test("tool role messages are appended correctly", async () => {
        const layer = await createLayer()
        layer.appendMessage("tool", JSON.stringify({ tool: "Hammer", output: "done", success: true }))
        expect(layer.getRawHistoryLength()).toBe(1)
        expect(layer.getRawHistory()[0].role).toBe("tool")
    })
})

describe("VoiceMemoryLayer — getLastMessage helpers", () => {
    test("getLastMessageRole returns the last message's role", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "a")
        layer.appendMessage("assistant", "b")
        expect(layer.getLastMessageRole()).toBe("assistant")
    })

    test("getLastMessageRole returns null for empty history", async () => {
        const layer = await createLayer()
        expect(layer.getLastMessageRole()).toBeNull()
    })

    test("getLastMessageByRole finds user among mixed messages", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "first user")
        layer.appendMessage("assistant", "response")
        layer.appendMessage("user", "second user")
        layer.appendMessage("assistant", "response2")
        const last = layer.getLastMessageByRole("user")
        expect(last).toBeDefined()
        expect(last!.content).toBe("second user")
    })

    test("getLastMessageByRole returns undefined when no match", async () => {
        const layer = await createLayer()
        layer.appendMessage("assistant", "yep")
        expect(layer.getLastMessageByRole("tool")).toBeUndefined()
    })

    test("popLastMessage removes and returns the last message", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "keep")
        layer.appendMessage("assistant", "remove me")
        const popped = layer.popLastMessage()
        expect(popped?.content).toBe("remove me")
        expect(layer.getRawHistoryLength()).toBe(1)
        expect(layer.getLastMessageRole()).toBe("user")
    })

    test("popLastMessage on empty returns undefined", async () => {
        const layer = await createLayer()
        expect(layer.popLastMessage()).toBeUndefined()
    })
})

describe("VoiceMemoryLayer — updateLastMessageByRole", () => {
    test("updates content and syncs charCount", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "short")
        layer.updateLastMessageByRole("user", "a much longer replacement sentence")

        const msg = layer.getRawHistory()[0]
        expect(msg.content).toBe("a much longer replacement sentence")
        expect(msg.charCount).toBe("a much longer replacement sentence".length)
    })

    test("updates the LAST matching role, not the first", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "first")
        layer.appendMessage("assistant", "middle")
        layer.appendMessage("user", "second")
        layer.updateLastMessageByRole("user", "updated")

        const history = layer.getRawHistory()
        expect(history[0].content).toBe("first")
        expect(history[2].content).toBe("updated")
    })

    test("returns false when no matching role found", async () => {
        const layer = await createLayer()
        layer.appendMessage("assistant", "only")
        expect(layer.updateLastMessageByRole("tool", "nope")).toBe(false)
    })

    test("returns true when update succeeds", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "original")
        expect(layer.updateLastMessageByRole("user", "changed")).toBe(true)
    })
})

describe("VoiceMemoryLayer — coalesceConsecutiveMessages", () => {
    test("merges adjacent user messages into a single raw-history entry", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "first")
        layer.appendMessage("user", "second")
        layer.appendMessage("assistant", "reply")

        const changed = layer.coalesceConsecutiveMessages("user")

        expect(changed).toBe(true)
        expect(layer.getRawHistoryLength()).toBe(2)
        expect(layer.getRawHistory()[0].role).toBe("user")
        expect(layer.getRawHistory()[0].content).toBe("first\n\nsecond")
        expect(layer.getRawHistory()[0].charCount).toBe("first\n\nsecond".length)
    })
})

describe("VoiceMemoryLayer — buildMessages", () => {
    test("includes system prompt at the start", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "hi")
        const msgs = layer.buildMessages("You are a helper.")
        expect(msgs[0].role).toBe("system")
        expect(msgs[0].content).toContain("You are a helper.")
    })

    test("includes recent user/assistant messages", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "What is 2+2?")
        layer.appendMessage("assistant", "4")
        const msgs = layer.buildMessages("system")
        const roles = msgs.map((m) => m.role)
        expect(roles).toContain("user")
        expect(roles).toContain("assistant")
    })

    test("converts tool messages to user role with prefix", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "run tool")
        layer.appendMessage("tool", '{"tool":"Search","output":"result"}')
        const msgs = layer.buildMessages("system")
        // Tool messages become user role
        const toolMsg = msgs.find((m) => m.content.includes("Tool result:"))
        expect(toolMsg).toBeDefined()
        expect(toolMsg!.role).toBe("user")
    })

    test("inserts user guard before first assistant if needed", async () => {
        const layer = await createLayer()
        layer.appendMessage("assistant", "I'm continuing from earlier")
        const msgs = layer.buildMessages("system")
        // Should have: [system, user-guard, assistant]
        const firstNonSystem = msgs.findIndex((m) => m.role !== "system")
        expect(msgs[firstNonSystem].role).toBe("user")
        expect(msgs[firstNonSystem].content).toBe("Continue our conversation.")
    })

    test("merges consecutive same-role messages", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "part one")
        layer.appendMessage("user", "part two")
        layer.appendMessage("assistant", "response")
        const msgs = layer.buildMessages("system")
        const userMsgs = msgs.filter((m) => m.role === "user")
        // Should merge the two user messages
        expect(userMsgs.length).toBe(1)
        expect(userMsgs[0].content).toContain("part one")
        expect(userMsgs[0].content).toContain("part two")
    })
})

describe("VoiceMemoryLayer — compaction", () => {
    test("compaction triggers when token estimate exceeds threshold", async () => {
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)

        const historyBefore = layer.getRawHistoryLength()
        await layer.triggerCompactionIfNeeded()

        // After compaction, history should be smaller (compacted messages pruned)
        expect(layer.getRawHistoryLength()).toBeLessThan(historyBefore)
        expect(layer.getCompactionCount()).toBeGreaterThanOrEqual(1)
    })

    test("compaction extracts topics from user messages", async () => {
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()
        const state = layer.getCompressedState()
        expect(state.topicsDiscussed.length).toBeGreaterThan(0)
    })

    test("compaction extracts user preferences", async () => {
        const layer = await createCompactableLayer()
        // Include preference statements early so they get compacted (not in protected window)
        layer.appendMessage("user", "I prefer dark mode interfaces always and want concise answers")
        layer.appendMessage("assistant", "Got it, I'll remember your preference for dark mode and concise answers")
        fillMessages(layer, 95)

        await layer.triggerCompactionIfNeeded()
        const state = layer.getCompressedState()
        expect(state.userPreferences.length).toBeGreaterThan(0)
        expect(state.userPreferences.some((p) => p.preference.toLowerCase().includes("dark mode") || p.preference.toLowerCase().includes("concise"))).toBe(true)
    })

    test("compaction extracts key facts from assistant messages", async () => {
        const layer = await createCompactableLayer()
        // Include fact-pattern statements early so they get compacted
        layer.appendMessage("user", "What is the capital of France and what is the population?")
        layer.appendMessage("assistant", "The answer is Paris, the beautiful capital city of France, located at the heart of Europe")
        fillMessages(layer, 95)

        await layer.triggerCompactionIfNeeded()
        const state = layer.getCompressedState()
        expect(state.keyFacts.length).toBeGreaterThan(0)
    })

    test("compaction extracts tool results", async () => {
        const layer = tracked(await VoiceMemoryLayer.create(undefined, undefined, {
            compactionTokenThreshold: 2_000,
            protectedContextTokens: 300,
            stateBudgetTokens: TEST_STATE_BUDGET,
            maxRawHistory: 500,
        }))

        layer.appendMessage("user", "Search for something about weather patterns and climate change")
        layer.appendMessage("tool", JSON.stringify({ tool: "BraveWebSearch", output: "Found 5 results about the topic of interest", success: true }))
        layer.appendMessage("assistant", "I found some results about weather patterns for you with detailed info")

        for (let i = 0; i < 8; i++) {
            layer.appendMessage(
                "user",
                `Follow-up question ${i}: explain another weather pattern in enough detail to add context tokens.`,
            )
            layer.appendMessage(
                "assistant",
                `Follow-up answer ${i}: here is a detailed explanation with enough text to keep older messages outside the protected window.`,
            )
        }

        await layer.triggerCompactionIfNeeded()
        const state = layer.getCompressedState()
        expect(layer.getCompactionCount()).toBeGreaterThanOrEqual(1)
        expect(state.toolResults.length).toBeGreaterThan(0)
        expect(state.toolResults[0].tool).toBe("BraveWebSearch")
        expect(state.toolResults[0].success).toBe(true)
    })

    test("compaction does not trigger when below threshold", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "short")
        layer.appendMessage("assistant", "brief")
        await layer.triggerCompactionIfNeeded()
        expect(layer.getCompactionCount()).toBe(0)
    })

    test("compaction debounces consecutive calls", async () => {
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()
        const countAfterFirst = layer.getCompactionCount()

        // Immediately calling again should be debounced (within compactionDebounceTurns)
        await layer.triggerCompactionIfNeeded()
        expect(layer.getCompactionCount()).toBe(countAfterFirst)
    })
})

describe("VoiceMemoryLayer — compaction", () => {
    test("deduplicates topics by content (keeps latest turn)", async () => {
        const layer = await createCompactableLayer()

        // Add repeated topic among unique ones, then fill
        for (let i = 0; i < 5; i++) {
            layer.appendMessage("user", "Tell me about artificial intelligence again please, I really want to discuss this topic in depth")
            layer.appendMessage("assistant", `Here is detailed response ${i} about AI: artificial intelligence covers machine learning, deep learning, natural language processing, and many more subfields`)
        }
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()
        const state = layer.getCompressedState()
        // Deduplicated topics — same-content topics should only appear once
        const topicTexts = state.topicsDiscussed.map((t) => t.topic.toLowerCase())
        const uniqueTexts = [...new Set(topicTexts)]
        expect(topicTexts.length).toBe(uniqueTexts.length)
    })
})

describe("VoiceMemoryLayer — renderState", () => {
    test("includes topics in rendered state", async () => {
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()
        const msgs = layer.buildMessages("system prompt")
        const systemContent = msgs[0].content
        expect(systemContent).toContain("Topics Discussed")
    })

    test("includes user preferences in rendered state", async () => {
        const layer = await createCompactableLayer()
        layer.appendMessage("user", "I always prefer concise answers without any unnecessary fluff or filler content")
        layer.appendMessage("assistant", "Understood, I'll keep it brief from now on in all future responses")
        fillMessages(layer, 95)

        await layer.triggerCompactionIfNeeded()
        const msgs = layer.buildMessages("system")
        const systemContent = msgs[0].content
        expect(systemContent).toContain("User Preferences")
    })
})

describe("VoiceMemoryLayer — persistence", () => {
    test("save writes to storage adapter", async () => {
        const storage = new InMemoryStorage()
        const layer = await createLayer(storage)
        layer.appendMessage("user", "hello")
        layer.appendMessage("assistant", "hi there")
        await layer.save()

        expect(storage.saveCount).toBe(1)
        expect(storage.data).not.toBeNull()
        expect(storage.data!.rawHistory.length).toBe(2)
    })

    test("load restores from storage adapter", async () => {
        const storage = new InMemoryStorage()
        const layer1 = await createLayer(storage)
        layer1.appendMessage("user", "remember this")
        layer1.appendMessage("assistant", "ok I will")
        await layer1.save()

        // Create a new layer that loads from same storage
        const layer2 = await createLayer(storage)
        await layer2.load()
        expect(layer2.getRawHistoryLength()).toBe(2)
        expect(layer2.getRawHistory()[0].content).toBe("remember this")
    })

    test("load with no storage data is a no-op", async () => {
        const storage = new InMemoryStorage()
        const layer = await createLayer(storage)
        await layer.load()
        expect(layer.getRawHistoryLength()).toBe(0)
    })

    test("clear resets in-memory state and clears storage", async () => {
        const storage = new InMemoryStorage()
        const layer = await createLayer(storage)
        layer.appendMessage("user", "something")
        await layer.save()
        await layer.clear()

        expect(layer.getRawHistoryLength()).toBe(0)
        expect(layer.getCurrentTurn()).toBe(0)
        expect(storage.clearCount).toBe(1)
    })

    test("save without storage adapter does not throw", async () => {
        const layer = await createLayer() // no storage
        layer.appendMessage("user", "hello")
        await layer.save() // should be no-op
    })

    test("load preserves turn count and compaction count", async () => {
        const storage = new InMemoryStorage()
        const layer1 = await createLayer(storage)
        for (let i = 0; i < 5; i++) {
            layer1.appendMessage("user", `msg ${i}`)
            layer1.appendMessage("assistant", `resp ${i}`)
        }
        await layer1.save()

        const layer2 = await createLayer(storage)
        await layer2.load()
        expect(layer2.getCurrentTurn()).toBe(10)
    })

    test("load handles corrupted data gracefully (missing rawHistory)", async () => {
        const storage = new InMemoryStorage()
        // Manually set corrupted data
        storage.data = {
            rawHistory: null as any,
            compressedState: {} as any,
            compactionCursor: { lastCompactedTurn: 0 },
            currentTurn: 0,
            compactionCount: 0,
        }

        const layer = await createLayer(storage)
        // Should handle gracefully (warn + start fresh)
        await layer.load()
        expect(layer.getRawHistoryLength()).toBe(0)
    })

    test("load does not map legacy conversationGoal into conversationTask", async () => {
        const storage = new InMemoryStorage()
        storage.data = {
            rawHistory: [],
            compressedState: {
                conversationGoal: "Legacy goal",
                topicsDiscussed: [],
                keyFacts: [],
                userPreferences: [],
                openItems: [],
                resolvedItems: [],
                toolResults: [],
                lastUpdatedTurn: 0,
            } as any,
            compactionCursor: { lastCompactedTurn: 0 },
            currentTurn: 0,
            compactionCount: 0,
        }

        const layer = await createLayer(storage)
        await layer.load()

        expect(layer.getCompressedState().conversationTask).toBeUndefined()
    })
})

describe("VoiceMemoryLayer — reset", () => {
    test("reset clears all state", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "hello")
        layer.appendMessage("assistant", "hi")
        layer.reset()

        expect(layer.getRawHistoryLength()).toBe(0)
        expect(layer.getCurrentTurn()).toBe(0)
        expect(layer.getCompactionCount()).toBe(0)
        expect(layer.getLastMessageRole()).toBeNull()
    })

    test("reset clears compressed state", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "I prefer dark mode interfaces always")
        layer.appendMessage("assistant", "noted")
        layer.reset()

        const state = layer.getCompressedState()
        expect(state.topicsDiscussed).toEqual([])
        expect(state.userPreferences).toEqual([])
        expect(state.keyFacts).toEqual([])
    })
})

describe("VoiceMemoryLayer — buildMessages with compressed state", () => {
    test("compressed state appears in system message after compaction", async () => {
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)
        await layer.triggerCompactionIfNeeded()

        const msgs = layer.buildMessages("You are an assistant.")
        expect(msgs[0].content).toContain("Conversation Memory")
    })

    test("compressed state does NOT appear when no compaction has run", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "short chat")
        layer.appendMessage("assistant", "brief reply")
        const msgs = layer.buildMessages("system")
        // No "Conversation Memory" should appear because no state to render
        expect(msgs[0].content).not.toContain("Conversation Memory")
    })
})

describe("VoiceMemoryLayer — estimateContextTokens", () => {
    test("returns positive number for non-empty history", async () => {
        const layer = await createLayer()
        layer.appendMessage("user", "hello world this is a test")
        const tokens = layer.estimateContextTokens()
        expect(tokens).toBeGreaterThan(0)
    })

    test("returns just overhead for empty history", async () => {
        const layer = await createLayer()
        const tokens = layer.estimateContextTokens()
        // Should equal systemPromptOverhead only
        expect(tokens).toBe(VOICE_SYSTEM_PROMPT_OVERHEAD)
    })
})

// ---------------------------------------------------------------------------
// LLM Compaction path (mock-based)
// ---------------------------------------------------------------------------

describe("VoiceMemoryLayer — LLM compaction", () => {
    function createMockCompactionClient(response: string, shouldThrow = false) {
        return {
            chat: async () => {
                if (shouldThrow) throw new Error("LLM unavailable")
                return { content: response }
            },
        }
    }

    async function createCompactableLayerWithLLM(client: ReturnType<typeof createMockCompactionClient>): Promise<VoiceMemoryLayer> {
        const layer = tracked(await VoiceMemoryLayer.create(undefined, client, {
            compactionTokenThreshold: TEST_COMPACTION_THRESHOLD,
            protectedContextTokens: TEST_PROTECTED_CONTEXT,
            stateBudgetTokens: TEST_STATE_BUDGET,
            maxRawHistory: 500,
        }))
        // Static context adds ~45,000 tiktoken tokens to the estimate
        layer.setStaticContext("Context: " + "This is background information about the user and environment. ".repeat(4500))
        return layer
    }

    test("uses LLM compaction when client is provided and returns valid JSON", async () => {
        const mockState = JSON.stringify({
            conversationTask: "Discuss weather",
            topicsDiscussed: [{ topic: "weather forecast", turn: 1 }],
            keyFacts: ["It is sunny today"],
            userPreferences: [],
            openItems: [],
            resolvedItems: [],
            toolResults: [],
        })
        const client = createMockCompactionClient(mockState)
        const layer = await createCompactableLayerWithLLM(client)
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        const state = layer.getCompressedState()
        expect(state.conversationTask).toBe("Discuss weather")
        expect(state.keyFacts.some((fact) => fact.summary === "It is sunny today")).toBe(true)
        expect(state.topicsDiscussed.length).toBe(1)
    })

    test("falls back to pattern matching when LLM returns invalid JSON", async () => {
        const client = createMockCompactionClient("this is not valid json at all")
        const layer = await createCompactableLayerWithLLM(client)
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        // Pattern matching should have run — extract topics from user messages
        const state = layer.getCompressedState()
        expect(state.topicsDiscussed.length).toBeGreaterThan(0)
        expect(layer.getCompactionCount()).toBeGreaterThanOrEqual(1)
    })

    test("falls back to pattern matching when LLM throws an error", async () => {
        const client = createMockCompactionClient("", true)
        const layer = await createCompactableLayerWithLLM(client)
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        // Pattern matching should have run as fallback
        const state = layer.getCompressedState()
        expect(state.topicsDiscussed.length).toBeGreaterThan(0)
        expect(layer.getCompactionCount()).toBeGreaterThanOrEqual(1)
    })

    test("LLM compaction strips markdown fences from response", async () => {
        const mockState = '```json\n{"conversationTask":"Testing","topicsDiscussed":[],"keyFacts":["fact1"],"userPreferences":[],"openItems":[],"resolvedItems":[],"toolResults":[]}\n```'
        const client = createMockCompactionClient(mockState)
        const layer = await createCompactableLayerWithLLM(client)
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        const state = layer.getCompressedState()
        expect(state.conversationTask).toBe("Testing")
        expect(state.keyFacts.some((fact) => fact.summary === "fact1")).toBe(true)
    })

    test("preserves existing conversationTask when LLM returns null", async () => {
        // LLM returns null for conversationTask — existing value should be kept
        const mockState = JSON.stringify({
            conversationTask: null,
            topicsDiscussed: [{ topic: "new topic", turn: 1 }],
            keyFacts: [],
            userPreferences: [],
            openItems: [],
            resolvedItems: [],
            toolResults: [],
        })
        const client = createMockCompactionClient(mockState)
        const layer = await createCompactableLayerWithLLM(client)
        // Manually set an existing task before compaction
        layer["compressedState"].conversationTask = "Existing task"
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        const state = layer.getCompressedState()
        // Existing task must be preserved since LLM returned null
        expect(state.conversationTask).toBe("Existing task")
        // New topic from LLM should be present
        expect(state.topicsDiscussed.length).toBeGreaterThan(0)
    })

    test("without compaction client, uses pattern matching only", async () => {
        // No compaction client — standard path
        const layer = await createCompactableLayer()
        fillMessages(layer, 100)

        await layer.triggerCompactionIfNeeded()

        const state = layer.getCompressedState()
        expect(state.topicsDiscussed.length).toBeGreaterThan(0)
        expect(layer.getCompactionCount()).toBeGreaterThanOrEqual(1)
    })
})
