/**
 * Tests for packages/utils/voice-agent-constants.ts
 *
 * Smoke tests to verify all exports exist and have sane types/ranges.
 */
import { describe, expect, test } from "vitest"
import {
    DASHSCOPE_WS_URL,
    ASR_SAMPLE_RATE,
    ASR_CHUNK_INTERVAL_MS,
    ASR_SPEECH_NOISE_THRESHOLD,
    ASR_MAX_SENTENCE_SILENCE_MS,
    MOBILE_ASR_SPEECH_NOISE_THRESHOLD,
    ASR_LANGUAGE_HINTS,
    CARTESIA_WS_URL,
    CARTESIA_VERSION,
    CARTESIA_MODEL_ID,
    CARTESIA_VOICE_ID,
    CARTESIA_OUTPUT_FORMAT,
    PLAYBACK_SAMPLE_RATE,
    LLM_TOKEN_BUFFER_THRESHOLD,
    VOICE_AGENT_SYSTEM_PROMPT
} from "../src/config"
import {
    BARE_ASSIGNMENT_STYLE_ARGUMENT_RULE,
    SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES,
    WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES,
} from "../src/tool-call-prompts"
import {
    VOICE_LLM_MAX_TOKENS,
    VOICE_LLM_TEMPERATURE,
} from "../src/config"

// ---------------------------------------------------------------------------
// ASR constants
// ---------------------------------------------------------------------------

describe("Voice agent constants — ASR", () => {
    test("DASHSCOPE_WS_URL is a wss:// URL", () => {
        expect(DASHSCOPE_WS_URL).toMatch(/^wss:\/\//)
    })

    test("ASR_SAMPLE_RATE is 16kHz", () => {
        expect(ASR_SAMPLE_RATE).toBe(16_000)
    })

    test("ASR_CHUNK_INTERVAL_MS is a positive number", () => {
        expect(ASR_CHUNK_INTERVAL_MS).toBeGreaterThan(0)
    })

    test("ASR_SPEECH_NOISE_THRESHOLD is in [-1, 1]", () => {
        expect(ASR_SPEECH_NOISE_THRESHOLD).toBeGreaterThanOrEqual(-1)
        expect(ASR_SPEECH_NOISE_THRESHOLD).toBeLessThanOrEqual(1)
    })

    test("ASR_MAX_SENTENCE_SILENCE_MS is a positive number", () => {
        expect(ASR_MAX_SENTENCE_SILENCE_MS).toBeGreaterThan(0)
    })

    test("MOBILE overrides are more lenient than desktop", () => {
        // Mobile noise threshold should be lower (less aggressive)
        expect(MOBILE_ASR_SPEECH_NOISE_THRESHOLD).toBeLessThanOrEqual(
            ASR_SPEECH_NOISE_THRESHOLD
        )
    })

    test("ASR_LANGUAGE_HINTS is a non-empty array", () => {
        expect(ASR_LANGUAGE_HINTS.length).toBeGreaterThan(0)
        expect(ASR_LANGUAGE_HINTS).toContain("en")
    })
})

// ---------------------------------------------------------------------------
// TTS / Cartesia constants
// ---------------------------------------------------------------------------

describe("Voice agent constants — TTS", () => {
    test("CARTESIA_WS_URL is a wss:// URL", () => {
        expect(CARTESIA_WS_URL).toMatch(/^wss:\/\//)
    })

    test("CARTESIA_VERSION is a date string", () => {
        expect(CARTESIA_VERSION).toMatch(/^\d{4}-\d{2}-\d{2}$/)
    })

    test("CARTESIA_MODEL_ID is a non-empty string", () => {
        expect(CARTESIA_MODEL_ID.length).toBeGreaterThan(0)
    })

    test("CARTESIA_VOICE_ID is a UUID-like string", () => {
        expect(CARTESIA_VOICE_ID).toMatch(
            /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
        )
    })

    test("CARTESIA_OUTPUT_FORMAT specifies PCM at PLAYBACK_SAMPLE_RATE", () => {
        expect(CARTESIA_OUTPUT_FORMAT.container).toBe("raw")
        expect(CARTESIA_OUTPUT_FORMAT.encoding).toBe("pcm_s16le")
        expect(CARTESIA_OUTPUT_FORMAT.sample_rate).toBe(PLAYBACK_SAMPLE_RATE)
    })

    test("PLAYBACK_SAMPLE_RATE matches Cartesia output", () => {
        expect(PLAYBACK_SAMPLE_RATE).toBe(CARTESIA_OUTPUT_FORMAT.sample_rate)
    })
})

// ---------------------------------------------------------------------------
// LLM / orchestration constants
// ---------------------------------------------------------------------------

describe("Voice agent constants — LLM & orchestration", () => {
    test("LLM_TOKEN_BUFFER_THRESHOLD is a small positive number", () => {
        expect(LLM_TOKEN_BUFFER_THRESHOLD).toBeGreaterThan(0)
        expect(LLM_TOKEN_BUFFER_THRESHOLD).toBeLessThan(100)
    })

    test("VOICE_LLM_MAX_TOKENS is reasonable", () => {
        expect(VOICE_LLM_MAX_TOKENS).toBeGreaterThan(0)
        expect(VOICE_LLM_MAX_TOKENS).toBeLessThanOrEqual(8192)
    })

    test("VOICE_LLM_TEMPERATURE is between 0 and 2", () => {
        expect(VOICE_LLM_TEMPERATURE).toBeGreaterThanOrEqual(0)
        expect(VOICE_LLM_TEMPERATURE).toBeLessThanOrEqual(2)
    })

    test("VOICE_AGENT_SYSTEM_PROMPT is a non-empty string", () => {
        expect(VOICE_AGENT_SYSTEM_PROMPT.length).toBeGreaterThan(0)
    })

    test("VOICE_AGENT_SYSTEM_PROMPT mentions concise/conversational", () => {
        const lower = VOICE_AGENT_SYSTEM_PROMPT.toLowerCase()
        expect(lower).toContain("concise")
    })

    test("VOICE_AGENT_SYSTEM_PROMPT includes tool-call protocol", () => {
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain('---tool---')
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain('---bash---')
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'Use ---tool--- for listed tools, ---bash--- for one-shot shell-native workflows'
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'Bash is not a registered tool name. Never start a tool payload with Bash --command ...; use the ---bash--- header instead.'
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'exactly one structured control block'
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'For registered tools, follow the exact CLI usage shown in the tool section.'
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'Use positional arguments only when the usage shows <param>; otherwise use --flag value.'
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).not.toContain('--flag=value')
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            BARE_ASSIGNMENT_STYLE_ARGUMENT_RULE.replace(
                ' unless that exact syntax is explicitly shown in the tool section.',
                '',
            )
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'exactly one structured control block',
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'Never emit multiple structured control blocks in a single response.',
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(
            'Only call tools that are explicitly listed below.',
        )
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES)
        expect(VOICE_AGENT_SYSTEM_PROMPT).toContain(SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES)
    })
})
