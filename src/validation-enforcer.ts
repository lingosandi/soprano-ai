/**
 * BaseValidationEnforcer — shared error-handling infrastructure for
 * any LLM agent loop (Hammer CLI, Tauri voice agent, etc.).
 *
 * Provides:
 *  • `surfaceError` / `surfaceWarning` → conversation injection
 *  • `handleValidationError` (run-line/Zod parse failures)
 *  • `handleApiError` (rate-limit, 5xx, etc.)
 *
 * Subclasses add domain-specific enforcement (skills-first policy,
 * TTS error announcements, etc.).
 */

// ── ConversationSink ─────────────────────────────────────────────────
// Minimal interface for injecting messages into any conversation store.
// Hammer's ConversationManager and the voice agent's ChatMessage[] both
// satisfy this with a trivial adapter.

export interface ConversationSink {
    addMessage(
        role: "user" | "assistant" | "system" | "tool",
        content: string,
    ): Promise<void>
}

export function createConversationSink(
    appendMessage: (
        role: "user" | "assistant" | "system" | "tool",
        content: string,
    ) => void | Promise<void>,
): ConversationSink {
    return {
        addMessage: async (role, content) => {
            await appendMessage(role, content)
        },
    }
}

// ── Result returned to the caller ────────────────────────────────────

export interface EnforcerResult {
    /** Error message surfaced to the agent as a user-visible message. */
    errorMessage: string
    /** JSON-serialised `{ success, error, data }` suitable for `lastToolResult`. */
    lastToolResult: string
}

type ValidationErrorWithRawContent = Error & {
    rawContent?: string
}

const VALIDATION_ERROR_PREFIX = "VALIDATION_ERROR: "

function hasValidationErrorPrefix(message: string): boolean {
    return message.startsWith(VALIDATION_ERROR_PREFIX)
}

function stripValidationErrorPrefix(message: string): string {
    return hasValidationErrorPrefix(message)
        ? message.slice(VALIDATION_ERROR_PREFIX.length)
        : message
}

function buildStepValidationSurfaceMessage(message: string): string {
    const details = stripValidationErrorPrefix(message)

    if (hasValidationErrorPrefix(message)) {
        return details
    }

    return `Validation error: ${details}. Reply with exactly one executable control block only. Put the standalone slug header on its own line and put the payload on the following line(s). Do not discuss the control syntax.`
}

function buildFatalSurfaceMessage(message: string): string {
    return `Fatal agent error: ${message}`
}

// ── Base class ───────────────────────────────────────────────────────

export abstract class BaseValidationEnforcer {
    protected sink: ConversationSink

    constructor(sink: ConversationSink) {
        this.sink = sink
    }

    // ── Missing tool call warning ────────────────────────────────────

    /**
     * Warn the agent that it sent a non-terminal response without any
     * executable control block — this wastes an action turn.
     */
    async handleMissingToolCall(actionCount: number): Promise<void> {
        this.logWarning(
            `No executable control block in response — action ${actionCount} was wasted.`,
        )

        await this.surfaceWarning(
            `Your response had no executable control block and no final ---bash--- control block whose payload was exit 0 or exit 1. ` +
            `This wasted an action turn. Reply with one concrete next action and do not explain the protocol back to the system.`,
        )
    }

    // ── Core: handle a validation error ──────────────────────────────

    /**
    * Handle an LLM response validation error (shell-line parse, Zod, etc.).
     *
     *  - surfaces a structured corrective error as a user message
     */
    async handleValidationError(
        error: Error,
    ): Promise<EnforcerResult> {
        this.logWarning(`Validation error: ${error.message}`)

        await this.surfaceValidationContext(error)

        const errorMessage = buildStepValidationSurfaceMessage(error.message)

        return this.surfaceError(errorMessage)
    }

    /**
     * Handle a non-validation LLM API error (rate-limit, 5xx, etc.).
     * Surfaces the message to the agent so it can retry on the next turn.
     */
    async handleApiError(error: Error): Promise<EnforcerResult> {
        this.logError(`LLM API error: ${error.message}`)
        return this.surfaceError(`LLM API error: ${error.message}`)
    }

    /**
     * Handle an unrecoverable runtime error that terminates the current run.
     * Surfaces the message through the same user-visible error channel as
     * validation and API failures.
     */
    async handleFatalError(error: Error | string): Promise<EnforcerResult> {
        const message = error instanceof Error ? error.message : error
        const errorMessage = buildFatalSurfaceMessage(message)

        this.logError(errorMessage)
        return this.surfaceError(errorMessage)
    }

    /**
     * Override to route warnings to a domain-specific logger.
     * Default: `console.warn`.
     */
    protected logWarning(message: string): void {
        console.warn(`[Agent] ⚠ ${message}`)
    }

    /**
     * Override to route errors to a domain-specific logger.
     * Default: `console.error`.
     */
    protected logError(message: string): void {
        console.error(`[Agent] ${message}`)
    }

    protected async surfaceValidationContext(error: Error): Promise<void> {
        void error
    }

    protected readValidationRawContent(error: Error): string | null {
        const rawContent = (error as ValidationErrorWithRawContent).rawContent
        if (typeof rawContent !== "string") {
            return null
        }

        const trimmedContent = rawContent.trim()
        return trimmedContent.length > 0 ? trimmedContent : null
    }

    // ── Internal: shared error/warning → user-message plumbing ──────

    protected async surfaceError(errorMessage: string): Promise<EnforcerResult> {
        const lastToolResult = JSON.stringify({
            success: false,
            error: errorMessage,
            data: null,
        })

        await this.sink.addMessage("user", `⚠️ ERROR: ${errorMessage}`)

        return { errorMessage, lastToolResult }
    }

    protected async surfaceWarning(warningMessage: string): Promise<void> {
        await this.sink.addMessage("user", `⚠️ WARNING: ${warningMessage}`)
    }
}
