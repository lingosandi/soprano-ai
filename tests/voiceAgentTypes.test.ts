/**
 * Tests for soprano-ai — state label & color helpers.
 */
import { describe, expect, test } from "vitest"
import {
    getAgentStateLabel,
    getAgentStateColor
} from "../src"
import type { AgentState } from "../src"

// ---------------------------------------------------------------------------
// getAgentStateLabel
// ---------------------------------------------------------------------------

describe("getAgentStateLabel", () => {
    const expectations: [AgentState, string][] = [
        ["idle", "Idle"],
        ["connecting", "Connecting…"],
        ["listening", "Listening…"],
        ["thinking", "Thinking…"],
        ["speaking", "Speaking…"]
    ]

    for (const [state, label] of expectations) {
        test(`'${state}' → '${label}'`, () => {
            expect(getAgentStateLabel(state)).toBe(label)
        })
    }

    test("returns 'Idle' for unknown state", () => {
        expect(getAgentStateLabel("unknown" as any)).toBe("Idle")
    })
})

// ---------------------------------------------------------------------------
// getAgentStateColor
// ---------------------------------------------------------------------------

describe("getAgentStateColor", () => {
    const allStates: AgentState[] = [
        "idle",
        "connecting",
        "listening",
        "thinking",
        "speaking"
    ]

    for (const state of allStates) {
        test(`'${state}' returns a hex color`, () => {
            const color = getAgentStateColor(state)
            expect(color).toMatch(/^#[0-9a-f]{6}$/i)
        })
    }

    test("each state has a unique color", () => {
        const colors = allStates.map(getAgentStateColor)
        const unique = new Set(colors)
        expect(unique.size).toBe(allStates.length)
    })

    test("unknown state returns idle color", () => {
        expect(getAgentStateColor("unknown" as any)).toBe(
            getAgentStateColor("idle")
        )
    })
})
