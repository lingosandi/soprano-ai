/**
 * Shared agent response parsing pipeline.
 *
 * Unifies the response parsing logic used by both Hammer CLI agent
 * (`BaseLLMProvider.parseAgentResponse`) and Magic webapp agent
 * (`useAgent.parseResponse`).
 *
 * Pipeline: raw content → structured control-segment
 * extraction → Zod validation
 */

import {
    DEFAULT_ALLOWED_RUN_TARGETS,
    parseStructuredAgentText,
    type RunInvocationTarget,
} from "./command-response-utils"
import { LLMResponseSchema } from "./schemas"
import type { LoopOutcome, ToolCall } from "./types"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ParsedAgentResponse {
    /** Zod-validated (or raw fallback) parsed response object. */
    parsed: Record<string, unknown>
    /** The single executable tool call selected for this turn, if any. */
    selectedToolCall?: ToolCall
    /** The outcome field ("continue", "success", "failure"). */
    outcome: LoopOutcome
    /** The reasoning string from the LLM. */
    reasoning: string
    /** Validation error message, if validation failed but raw data was usable. */
    validationError?: string
    /** Original raw content string. */
    raw: string
}

export interface ParseAgentResponseOptions {
    /**
     * If `true`, throw on Zod validation failure (Hammer behavior).
     * If `false`, fall back to raw parsed data on Zod failure (Magic behavior).
     * Default: `false`.
     */
    throwOnValidationError?: boolean
    /** Allow truncated structured control segments during recovery paths. */
    allowTruncatedRuns?: boolean
    /**
     * Custom error formatter called when `throwOnValidationError` is `true`.
     * Receives the Zod error and raw content, should return an Error to throw.
     */
    formatValidationError?: (zodError: unknown, rawContent: string) => Error
    /** Allowed structured control targets for this agent. Defaults to tool/bash. */
    allowedRunTargets?: readonly RunInvocationTarget[]
}

// ---------------------------------------------------------------------------
// Core parsing function
// ---------------------------------------------------------------------------

/**
 * Parse an LLM response string into a structured agent response.
 *
 * Shared pipeline:
 * 1. Extract prose + structured control segments
 * 2. Validate via `LLMResponseSchema` (Zod)
 * 3. Return structured result or null if extraction fails
 *
 * @returns Parsed response, or `null` if no structured control segments could be extracted.
 */
export function parseAgentResponse(
    content: string,
    options?: ParseAgentResponseOptions,
): ParsedAgentResponse | null {
    const {
        throwOnValidationError = false,
        allowTruncatedRuns = false,
        formatValidationError,
        allowedRunTargets = DEFAULT_ALLOWED_RUN_TARGETS,
    } = options ?? {}

    const result = parseStructuredAgentText(content, {
        allowTruncated: allowTruncatedRuns,
        allowedTargets: allowedRunTargets,
    })
    if (!result) return null

    const parsed: Record<string, unknown> = {
        reasoning: result.prose,
        outcome: result.outcome,
        ...(result.selectedToolCall ? { selectedToolCall: result.selectedToolCall } : {}),
    }

    let validationError: string | undefined
    let validated: Record<string, unknown>

    try {
        validated = LLMResponseSchema.parse(parsed) as Record<string, unknown>
    } catch (err: unknown) {
        if (throwOnValidationError) {
            if (formatValidationError) {
                throw formatValidationError(err, content)
            }
            throw err
        }
        // Lenient mode: fall back to raw parsed data
        validationError = err instanceof Error ? err.message : String(err)
        validated = parsed
    }

    const outcome = (validated.outcome ?? parsed.outcome ?? "continue") as ParsedAgentResponse["outcome"]
    const reasoning = String(validated.reasoning ?? parsed.reasoning ?? "")
    const selectedToolCall =
        (validated.selectedToolCall as ToolCall | undefined) ?? result.selectedToolCall

    return {
        parsed: validated,
        ...(selectedToolCall ? { selectedToolCall } : {}),
        outcome,
        reasoning,
        validationError,
        raw: content,
    }
}
