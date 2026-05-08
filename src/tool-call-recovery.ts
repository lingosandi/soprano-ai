/**
 * Shared tool-call recovery utilities for the structured control-segment protocol.
 *
 * Extracts truncation recovery, validation error formatting, and error
 * message templates so Hammer, Magic, Monoslides, Monospace, and the voice agent share one
 * response contract.
 */

import { parseAgentResponse, type ParsedAgentResponse, type ParseAgentResponseOptions } from "./agent-response-parser"
import { decodeEscapedShellText } from "./shell-escape-normalization"
import {
    containsStandaloneStructuredInvocationStart,
} from "./command-response-utils"
import {
    VALIDATION_FIX_REFERENCE,
    buildMultipleStructuredControlSegmentsValidationError,
    buildStructuredControlValidationError,
} from "./tool-call-prompts"
import { DEFAULT_ALLOWED_RUN_TARGETS, type RunInvocationTarget } from "./command-response-utils"

/** Error message sent to the LLM when its response was truncated mid-generation. */
export const ERROR_TRUNCATED_RESPONSE = `VALIDATION_ERROR: Your response was truncated before the final executable control block could be extracted.

Retry with normal prose and one final executable control block only. Put the standalone slug header on its own line and put the payload on the following line(s). Do not discuss the control syntax.`

// ---------------------------------------------------------------------------
// Error formatting
// ---------------------------------------------------------------------------

/**
 * Format a Zod validation error (or generic Error) into a human-readable string.
 *
 * Returns per-field details showing which parts are missing or have wrong types,
 * suitable for feeding back to the LLM so it can self-correct.
 */
export function formatZodValidationError(zodError: unknown): string {
    // Zod errors have an `issues` array
    if (
        zodError &&
        typeof zodError === "object" &&
        "issues" in zodError &&
        Array.isArray((zodError as any).issues)
    ) {
        return (zodError as any).issues
            .map((err: any) => {
                const path = (err.path ?? []).join(".") || "(root)"
                if (path === "outcome") {
                    let detail =
                        '  - terminal state: continue responses do not include a finish control block; only final responses use a ---bash--- control block with payload exit 0 or exit 1'
                    if (err.received !== undefined) {
                        detail += ` (you sent: ${JSON.stringify(err.received)})`
                    }
                    return detail
                }
                let detail = `  - ${path}: ${err.message}`
                if (err.received === undefined && err.code === "invalid_type") {
                    detail += " (field is MISSING from your response)"
                } else if (err.received !== undefined) {
                    detail += ` (you sent: ${JSON.stringify(err.received)})`
                }
                if (err.expected && err.code === "invalid_type") {
                    detail += ` — expected type: ${err.expected}`
                }
                return detail
            })
            .join("\n")
    }
    if (zodError instanceof Error) {
        return `Response parse error: ${zodError.message}`
    }
    return String(zodError)
}

/**
 * Build a complete VALIDATION_ERROR string with Zod details + fix reference.
 *
 * Suitable for throwing as an `Error.message` that gets sent back to the LLM.
 */
export function buildValidationErrorMessage(zodError: unknown): string {
    const details = formatZodValidationError(zodError)
    return buildStructuredControlValidationError(details)
}

/**
 * Build a "no structured control segment found" error message.
 *
 * Keep this message focused on the repair instruction instead of duplicating
 * the failed output inline.
 */
export function buildNoStructuredResponseFoundError(): string {
    return buildStructuredControlValidationError(
        'Your response did not end with an executable control block. Reply with normal prose and, if you are acting, one final standalone header block such as ---tool---, ---bash---, or ---background_bash---. Put the header on its own line and put the payload on the following line(s). Use ---bash--- with exit 0 or exit 1 only when finishing. Do not discuss the control syntax.',
    )
}

export {
    buildMultipleStructuredControlSegmentsValidationError,
    VALIDATION_FIX_REFERENCE,
}

// ---------------------------------------------------------------------------
// Recovery strategies
// ---------------------------------------------------------------------------

/**
 * Attempt to recover a parseable agent response from truncated LLM output
 * (when `finishReason === "length"`).
 *
 * Strategy:
 * 1. Try `parseAgentResponse` in lenient mode with truncated control-segment support
 * 2. Infer "continue" for non-terminal responses when control segments exist
 *
 * @returns A `ParsedAgentResponse`, or `null` if nothing could be extracted.
 */
export function recoverTruncatedResponse(
    content: string,
    options?: Pick<ParseResponseWithRecoveryOptions, "allowedRunTargets">,
): ParsedAgentResponse | null {
    const fullResult = parseAgentResponse(content, {
        throwOnValidationError: false,
        allowTruncatedRuns: true,
        allowedRunTargets: options?.allowedRunTargets,
    })
    const bashToolLogRecovery = recoverBashCommandFromToolLog(content, {
        allowedRunTargets: options?.allowedRunTargets,
    })

    if (fullResult) {
        if (shouldPreferBashToolLogRecovery(fullResult, bashToolLogRecovery)) {
            return bashToolLogRecovery
        }

        return fullResult
    }

    if (bashToolLogRecovery) return bashToolLogRecovery

    return null
}

function recoverBashCommandFromToolLog(
    content: string,
    options?: Pick<ParseResponseWithRecoveryOptions, "allowedRunTargets">,
): ParsedAgentResponse | null {
    const toolLogCommand = extractBashCommandFromToolLog(content)
    if (!toolLogCommand) {
        return null
    }

    return parseAgentResponse(
        `Recover the last bash command from the tool log.\n---bash---\n${toolLogCommand}`,
        {
            throwOnValidationError: false,
            allowTruncatedRuns: true,
            allowedRunTargets: options?.allowedRunTargets,
        },
    )
}

function extractBashCommandFromToolLog(content: string): string | null {
    const match = content.match(/\[TOOL_LOG:Bash:[^\]]+\]\s*\r?\nOPEN\s*\r?\n([\s\S]+?)(?=\r?\n\[(?:stderr|meta|exit)\])/i)
    if (!match?.[1]) {
        return null
    }

    const normalizedBlock = decodeEscapedShellText(match[1]).trim()
    const lines = normalizedBlock.split(/\r?\n/)
    if (lines.length === 0) {
        return null
    }

    lines[0] = lines[0]!.replace(/^\$\s+/, "")

    const command = lines.join("\n").trim()
    return command.length > 0 ? command : null
}

function shouldPreferBashToolLogRecovery(
    current: ParsedAgentResponse,
    recovered: ParsedAgentResponse | null,
): recovered is ParsedAgentResponse {
    if (!recovered) {
        return false
    }

    const currentToolCall = current.selectedToolCall
    if (!currentToolCall || currentToolCall.kind !== "bash") {
        return true
    }

    const currentCommand = currentToolCall.parameters.command
    const recoveredCommand = recovered.selectedToolCall?.parameters.command

    if (typeof currentCommand !== "string" || typeof recoveredCommand !== "string") {
        return false
    }

    return currentCommand !== recoveredCommand && (
        currentToolCall.truncated === true
        || /\n\[(?:TOOL_LOG|stderr|meta|exit):?/i.test(currentCommand)
        || /\n\[(?:stderr|meta|exit)\]/i.test(currentCommand)
    )
}

/**
 * Build a user-feedback message to send to the LLM when its response failed
 * to parse. Gives more actionable guidance than a generic "parse failed" message.
 *
 * @param content     The raw LLM content that failed to parse
 * @param zodError    Optional Zod validation error for field-specific feedback
 * @returns           A feedback string to send as a user message
 */
export function buildParseFeedback(_content: string, zodError?: unknown): string {
    // Check if there's Zod field-level detail
    if (zodError) {
        return buildValidationErrorMessage(zodError)
    }

    return buildNoStructuredResponseFoundError()
}

// ---------------------------------------------------------------------------
// Unified three-tier parsing cascade
// ---------------------------------------------------------------------------

export interface ParseResponseWithRecoveryOptions extends ParseAgentResponseOptions {
    /**
     * The LLM's `finishReason` string. When `"length"`, Tier 2 truncation
     * recovery is attempted.
     */
    finishReason?: string
}

function looksLikeStandaloneControlSegment(
    content: string,
    allowedRunTargets: readonly RunInvocationTarget[],
): boolean {
    return containsStandaloneStructuredInvocationStart(content, allowedRunTargets)
}

function getSelectedToolCallKey(result: ParsedAgentResponse): string | undefined {
    const selectedToolCall = result.selectedToolCall
    if (!selectedToolCall) {
        return undefined
    }

    if (
        selectedToolCall.kind === "bash"
        || selectedToolCall.kind === "background_bash"
    ) {
        const command = selectedToolCall.parameters.command
        return typeof command === "string" ? command : undefined
    }

    return selectedToolCall.rawInvocation ?? selectedToolCall.name
}

/**
 * Three-tier parsing cascade for LLM step responses.
 *
 * Shared by Magic (`use-agent.ts`) and Hammer (`BaseLLMProvider`).
 *
 *   Tier 1: `parseAgentResponse` — control-segment extraction + Zod validation
 *   Tier 2: `recoverTruncatedResponse` — if `finishReason === "length"`
 *
 * Tier 1 respects `throwOnValidationError`: if structured lines are found but
 * validation fails in strict mode, the error propagates immediately.
 *
 * @returns Parsed response, or `null` if nothing could be extracted.
 */
export function parseResponseWithRecovery(
    content: string,
    options?: ParseResponseWithRecoveryOptions,
): ParsedAgentResponse | null {
    const {
        finishReason,
        throwOnValidationError,
        formatValidationError,
        allowedRunTargets,
    } = options ?? {}
    const effectiveAllowedRunTargets = allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS

    const result = parseAgentResponse(content, {
        throwOnValidationError,
        formatValidationError,
        allowedRunTargets: effectiveAllowedRunTargets,
    })
    if (result) {
        const bashToolLogRecovery = recoverBashCommandFromToolLog(content, {
            allowedRunTargets: effectiveAllowedRunTargets,
        })

        if (shouldPreferBashToolLogRecovery(result, bashToolLogRecovery)) {
            return bashToolLogRecovery
        }

        if (finishReason === "length") {
            const truncated = recoverTruncatedResponse(content, {
                allowedRunTargets: effectiveAllowedRunTargets,
            })
            const currentToolCallKey = getSelectedToolCallKey(result)

            if (
                truncated &&
                (
                    getSelectedToolCallKey(truncated) !== currentToolCallKey
                )
            ) {
                return truncated
            }
        }

        return result
    }

    if (
        finishReason === "length"
        || looksLikeStandaloneControlSegment(content, effectiveAllowedRunTargets)
    ) {
        const truncated = recoverTruncatedResponse(content, {
            allowedRunTargets: effectiveAllowedRunTargets,
        })
        if (truncated) return truncated
    }

    return null
}
