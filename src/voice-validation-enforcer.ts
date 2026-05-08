/**
 * VoiceValidationEnforcer — voice-agent-specific subclass of
 * BaseValidationEnforcer.
 *
 * Adds three capabilities missing from the raw voice pipeline:
 *
 *  1. **Tool-call parse error feedback** — when streaming-tool-parser
 *     fails to parse a tool call, the error is injected back into the
 *     conversation so the LLM can self-correct on the next turn.
 *
 *  2. **LLM error feedback** — surfaces LLM API errors (429, 5xx) back
 *     into the conversation instead of silently dropping to idle.
 *
 * Adapter wiring is shared via `createConversationSink(...)`; this class
 * only owns voice-specific validation and logging behavior.
 */

import {
    BaseValidationEnforcer,
    type ConversationSink,
    type EnforcerResult,
} from "./validation-enforcer"
import { buildParseFeedback } from "./tool-call-recovery"

// ── VoiceValidationEnforcer ──────────────────────────────────────────


export class VoiceValidationEnforcer extends BaseValidationEnforcer {
    private onLog?: (msg: string) => void

    constructor(sink: ConversationSink, onLog?: (msg: string) => void) {
        super(sink)
        this.onLog = onLog
    }

    async handleToolCallParseError(
        error: Error,
        rawContent: string,
        zodError?: unknown,
    ): Promise<EnforcerResult> {
        this.logWarning(
            `Tool-call parse error: ${error.message} (raw: ${rawContent.slice(0, 200)})`,
        )

        const wrapped = new Error(
            buildParseFeedback(rawContent, zodError ?? error),
        )

        return this.handleValidationError(wrapped)
    }

    // ── LLM error feedback ───────────────────────────────────────────

    /**
     * Handle an LLM streaming error and surface it back to the agent.
     */
    async handleStreamError(error: Error): Promise<EnforcerResult> {
        this.logWarning(`LLM stream error: ${error.message}`)
        return this.surfaceError(`LLM stream error: ${error.message}`)
    }

    // ── Protected overrides ──────────────────────────────────────────



    protected override logWarning(message: string): void {
        this.onLog?.(`[VoiceEnforcer] ⚠ ${message}`)
    }

    protected override logError(message: string): void {
        this.onLog?.(`[VoiceEnforcer] ${message}`)
    }
}
