/**
 * Tests for soprano-ai
 *
 * Covers:
 *   - HttpMemoryAdapter: string constructor, lazy getter constructor,
 *     load (ok, 204/404, error, timeout), save (ok, error, res.ok check),
 *     clear (ok, error)
 *   - LocalStorageMemoryAdapter: load, save, clear, error handling
 */
import { describe, expect, test, beforeEach, afterEach } from "vitest"

import {
    HttpMemoryAdapter,
    LocalStorageMemoryAdapter,
} from "../src"
import type {
    PersistedVoiceMemory,
    VoiceFact,
    VoiceMessage,
    VoicePreference,
    VoiceTopic,
} from "../src"

// ===========================================================================
// Mock fetch infrastructure
// ===========================================================================

let originalFetch: typeof globalThis.fetch
let fetchCalls: Array<{ url: string; init?: RequestInit }> = []

function mockFetch(handler: (url: string, init?: RequestInit) => Promise<Response>) {
    ;(globalThis as any).fetch = async (input: any, init?: any) => {
        const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : (input as Request).url
        fetchCalls.push({ url, init })
        return handler(url, init)
    }
}

beforeEach(() => {
    originalFetch = globalThis.fetch
    fetchCalls = []
})

afterEach(() => {
    ;(globalThis as any).fetch = originalFetch
})

// ===========================================================================
// Test data
// ===========================================================================

const sampleMemory: PersistedVoiceMemory = {
    rawHistory: [
        { role: "user", content: "Hello", charCount: 5, timestamp: Date.now() },
        { role: "assistant", content: "Hi there!", charCount: 9, timestamp: Date.now() },
    ] as VoiceMessage[],
    compressedState: {
        topicsDiscussed: [{
            topic: "Greeting",
            canonical: "greeting",
            provenance: { source: "user", detail: "persisted" },
            turn: 1,
        }] satisfies VoiceTopic[],
        keyFacts: [{
            summary: "User prefers dark mode",
            canonical: "user prefers dark mode",
            provenance: { source: "assistant", detail: "persisted" },
            turn: 1,
        }] satisfies VoiceFact[],
        userPreferences: [{
            preference: "dark mode",
            canonical: "dark mode",
            provenance: { source: "user", detail: "persisted" },
            turn: 1,
        }] satisfies VoicePreference[],
        openItems: [],
        resolvedItems: [],
        toolResults: [],
        lastUpdatedTurn: 1,
    },
    compactionCursor: { lastCompactedTurn: 0 },
    currentTurn: 1,
    compactionCount: 0,
}

// ===========================================================================
// HttpMemoryAdapter — constructor
// ===========================================================================

describe("HttpMemoryAdapter — constructor", () => {
    test("accepts string URL", async () => {
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        const adapter = new HttpMemoryAdapter("http://127.0.0.1:9231")
        await adapter.load()

        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/voice-memory")
    })

    test("accepts lazy getter function", async () => {
        let port = 9231
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        const adapter = new HttpMemoryAdapter(() => `http://127.0.0.1:${port}`)
        await adapter.load()
        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/voice-memory")

        // Change port — next call should use new value
        port = 9232
        await adapter.load()
        expect(fetchCalls[1].url).toBe("http://127.0.0.1:9232/voice-memory")
    })

    test("accepts a custom HTTP path", async () => {
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        const adapter = new HttpMemoryAdapter("http://127.0.0.1:9231", "/custom-memory")
        await adapter.load()

        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/custom-memory")
    })
})

// ===========================================================================
// HttpMemoryAdapter — load
// ===========================================================================

describe("HttpMemoryAdapter — load", () => {
    const adapter = new HttpMemoryAdapter("http://127.0.0.1:9231")

    test("returns parsed memory on 200", async () => {
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        const result = await adapter.load()
        expect(result).not.toBeNull()
        expect(result!.rawHistory).toHaveLength(2)
        expect(result!.compressedState.keyFacts[0]?.summary).toBe("User prefers dark mode")
    })

    test("returns null on 204 empty memory", async () => {
        mockFetch(async () => new Response(null, { status: 204 }))

        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("returns null on legacy 404 empty memory", async () => {
        mockFetch(async () => new Response("Not Found", { status: 404 }))

        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("returns null on server error (500)", async () => {
        mockFetch(async () => new Response("Error", { status: 500 }))

        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("returns null on network error", async () => {
        mockFetch(async () => { throw new Error("ECONNREFUSED") })

        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("sends correct headers", async () => {
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        await adapter.load()
        expect(fetchCalls[0].init?.headers).toBeDefined()
    })

    test("uses GET method", async () => {
        mockFetch(async () => new Response(JSON.stringify(sampleMemory), { status: 200 }))

        await adapter.load()
        expect(fetchCalls[0].init?.method).toBe("GET")
    })
})

// ===========================================================================
// HttpMemoryAdapter — save
// ===========================================================================

describe("HttpMemoryAdapter — save", () => {
    const adapter = new HttpMemoryAdapter("http://127.0.0.1:9231")

    test("sends PUT request with JSON body", async () => {
        mockFetch(async () => new Response("OK", { status: 200 }))

        await adapter.save(sampleMemory)

        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/voice-memory")
        expect(fetchCalls[0].init?.method).toBe("PUT")
        const body = JSON.parse(fetchCalls[0].init?.body as string)
        expect(body.rawHistory).toHaveLength(2)
    })

    test("does not throw on success", async () => {
        mockFetch(async () => new Response("OK", { status: 200 }))

        // Should not throw
        await adapter.save(sampleMemory)
    })

    test("does not throw on server error (logs instead)", async () => {
        mockFetch(async () => new Response("Error", { status: 500 }))

        // Should not throw — logs error instead
        await adapter.save(sampleMemory)
    })

    test("does not throw on network error (logs instead)", async () => {
        mockFetch(async () => { throw new Error("ECONNREFUSED") })

        // Should not throw
        await adapter.save(sampleMemory)
    })

    test("uses the custom path for saves", async () => {
        mockFetch(async () => new Response("OK", { status: 200 }))

        const customAdapter = new HttpMemoryAdapter("http://127.0.0.1:9231", "/custom-memory")
        await customAdapter.save(sampleMemory)

        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/custom-memory")
        expect(fetchCalls[0].init?.method).toBe("PUT")
    })
})

// ===========================================================================
// HttpMemoryAdapter — clear
// ===========================================================================

describe("HttpMemoryAdapter — clear", () => {
    const adapter = new HttpMemoryAdapter("http://127.0.0.1:9231")

    test("sends DELETE request", async () => {
        mockFetch(async () => new Response("OK", { status: 200 }))

        await adapter.clear()

        expect(fetchCalls[0].url).toBe("http://127.0.0.1:9231/voice-memory")
        expect(fetchCalls[0].init?.method).toBe("DELETE")
    })

    test("does not throw on error", async () => {
        mockFetch(async () => { throw new Error("ECONNREFUSED") })

        // Should not throw
        await adapter.clear()
    })
})

// ===========================================================================
// LocalStorageMemoryAdapter
// ===========================================================================

// Mock localStorage for Bun test environment
let storage: Record<string, string> = {}

beforeEach(() => {
    storage = {}
    ;(globalThis as any).localStorage = {
        getItem: (key: string) => storage[key] ?? null,
        setItem: (key: string, value: string) => { storage[key] = value },
        removeItem: (key: string) => { delete storage[key] },
    }
})

afterEach(() => {
    delete (globalThis as any).localStorage
})

describe("LocalStorageMemoryAdapter — load", () => {
    test("returns null when no data stored", async () => {
        const adapter = new LocalStorageMemoryAdapter()
        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("returns parsed data when stored", async () => {
        storage["mjolno_voice_memory"] = JSON.stringify(sampleMemory)
        const adapter = new LocalStorageMemoryAdapter()
        const result = await adapter.load()
        expect(result).not.toBeNull()
        expect(result!.compressedState.keyFacts[0]?.summary).toBe("User prefers dark mode")
    })

    test("returns null on invalid JSON", async () => {
        storage["mjolno_voice_memory"] = "not json"
        const adapter = new LocalStorageMemoryAdapter()
        const result = await adapter.load()
        expect(result).toBeNull()
    })

    test("uses custom key", async () => {
        storage["custom_key"] = JSON.stringify(sampleMemory)
        const adapter = new LocalStorageMemoryAdapter("custom_key")
        const result = await adapter.load()
        expect(result).not.toBeNull()
    })
})

describe("LocalStorageMemoryAdapter — save", () => {
    test("saves data to localStorage", async () => {
        const adapter = new LocalStorageMemoryAdapter()
        await adapter.save(sampleMemory)
        expect(storage["mjolno_voice_memory"]).toBeDefined()
        const saved = JSON.parse(storage["mjolno_voice_memory"])
        expect(saved.compressedState.keyFacts[0]?.summary).toBe("User prefers dark mode")
    })

    test("uses custom key", async () => {
        const adapter = new LocalStorageMemoryAdapter("custom_key")
        await adapter.save(sampleMemory)
        expect(storage["custom_key"]).toBeDefined()
    })
})

describe("LocalStorageMemoryAdapter — clear", () => {
    test("removes data from localStorage", async () => {
        storage["mjolno_voice_memory"] = JSON.stringify(sampleMemory)
        const adapter = new LocalStorageMemoryAdapter()
        await adapter.clear()
        expect(storage["mjolno_voice_memory"]).toBeUndefined()
    })
})
