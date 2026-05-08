/**
 * VoiceMemoryStorage adapters — platform-specific persistence.
 *
 * Each adapter implements VoiceMemoryStorage from voice-memory.ts.
 * Import only the adapter you need in your app.
 */

import type { VoiceMemoryStorage, PersistedVoiceMemory } from "./voice-memory"

// ============================================================================
// LocalStorage adapter (Tauri webview, plain browser)
// ============================================================================

const DEFAULT_KEY = "mjolno_voice_memory"
const DEFAULT_HTTP_PATH = "/voice-memory"

/**
 * Persists voice memory to browser localStorage.
 *
 * Works in any browser context (Tauri webview, desktop browser).
 * Data survives page reloads and app restarts.
 */
export class LocalStorageMemoryAdapter implements VoiceMemoryStorage {
    private key: string

    constructor(key = DEFAULT_KEY) {
        this.key = key
    }

    async load(): Promise<PersistedVoiceMemory | null> {
        try {
            const raw = localStorage.getItem(this.key)
            if (!raw) return null
            return JSON.parse(raw) as PersistedVoiceMemory
        } catch (err) {
            console.warn("[VoiceMemory] Failed to load from localStorage:", err)
            return null
        }
    }

    async save(data: PersistedVoiceMemory): Promise<void> {
        try {
            localStorage.setItem(this.key, JSON.stringify(data))
        } catch (err) {
            console.error("[VoiceMemory] Failed to save to localStorage:", err)
        }
    }

    async clear(): Promise<void> {
        localStorage.removeItem(this.key)
    }
}

// ============================================================================
// HTTP adapter (browser → Bun server for disk persistence)
// ============================================================================

/**
 * Persists voice memory by calling a Bun server HTTP endpoint.
 *
 * Use this in the Tauri webview when you want actual JSON files on disk
 * rather than localStorage. Pair with the /voice-memory endpoints in
 * the Bun server (src-server/).
 */
export class HttpMemoryAdapter implements VoiceMemoryStorage {
    /** String or lazy getter — called each time so port discovery has time to resolve. */
    private getBaseUrl: () => string
    private path: string

    constructor(baseUrl: string | (() => string), path = DEFAULT_HTTP_PATH) {
        this.getBaseUrl = typeof baseUrl === "function" ? baseUrl : () => baseUrl
        this.path = path.startsWith("/") ? path : `/${path}`
    }

    private buildUrl(): string {
        return `${this.getBaseUrl()}${this.path}`
    }

    async load(): Promise<PersistedVoiceMemory | null> {
        try {
            const res = await fetch(this.buildUrl(), {
                method: "GET",
                headers: { "Content-Type": "application/json" },
                signal: AbortSignal.timeout(3_000),
            })
            if (res.status === 204) return null
            if (!res.ok) return null
            return (await res.json()) as PersistedVoiceMemory
        } catch {
            return null
        }
    }

    async save(data: PersistedVoiceMemory): Promise<void> {
        const url = this.buildUrl()
        try {
            const res = await fetch(url, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
                signal: AbortSignal.timeout(3_000),
            })
            if (!res.ok) {
                const body = await res.text().catch(() => "")
                console.error(`[VoiceMemory] Save failed (${res.status}): ${body}`)
            }
        } catch (err) {
            console.error("[VoiceMemory] Failed to save via HTTP:", err)
        }
    }

    async clear(): Promise<void> {
        try {
            await fetch(this.buildUrl(), {
                method: "DELETE",
                signal: AbortSignal.timeout(3_000),
            })
        } catch {
            // Ignore
        }
    }
}
