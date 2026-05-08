/**
 * StreamingToolParser — separates speech from structured control segments
 * during streaming. It normally parses tool calls at stream end, but if a
 * second control header appears it immediately seals the stream and keeps only
 * the first segment.
 */

import {
    DEFAULT_ALLOWED_RUN_TARGETS,
    containsStandaloneStructuredInvocationStart,
    parseStructuredAgentText,
    type RunInvocationTarget,
} from "./command-response-utils"
import { SUPPORTED_RUN_TARGETS } from "./run-command-registry"
import type { ToolCall } from "./types"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StreamingToolParserCallbacks {
    /** Fired for every token that is regular speech (not part of a control segment). */
    onSpeechToken?: (token: string) => void
    /** Fired when structured control segments have been parsed into tool calls. */
    onToolCall?: (selectedToolCall: ToolCall) => void
    /** Fired if structured command text was detected but could not be parsed. */
    onToolCallError?: (error: Error, rawCommandText: string) => void
    /** Informational log messages. */
    onLog?: (msg: string) => void
    /** Allowed control-segment targets for this parser. Defaults to tool/bash. */
    allowedRunTargets?: readonly RunInvocationTarget[]
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

export class StreamingToolParser {
    private speechBuffer = ""
    private segmentBuffer = ""
    private inSegment = false
    private _sealed = false
    private callbacks: StreamingToolParserCallbacks = {}

    /** The full raw content accumulated so far (speech + structured lines). */
    private fullContent = ""

    private static buildControlHeaderPattern(
        allowedTargets: readonly RunInvocationTarget[],
        flags: string,
    ): RegExp | null {
        const normalizedTargets = allowedTargets.filter((target): target is RunInvocationTarget =>
            SUPPORTED_RUN_TARGETS.includes(target),
        )

        if (normalizedTargets.length === 0) {
            return null
        }

        return new RegExp(
            `(^|\\n)([ \t]*)---(?:${normalizedTargets.join("|")})---(?:[ \t]*\\r?\\n[ \t]*|[ \t]+|$)`,
            flags,
        )
    }

    private static getControlHeaders(allowedTargets: readonly RunInvocationTarget[]): string[] {
        return allowedTargets
            .filter((target): target is RunInvocationTarget =>
                SUPPORTED_RUN_TARGETS.includes(target),
            )
            .map((target) => `---${target}---`)
    }

    private static findControlHeader(
        content: string,
        allowedTargets: readonly RunInvocationTarget[],
    ): { index: number } | null {
        const pattern = StreamingToolParser.buildControlHeaderPattern(
            allowedTargets,
            "i",
        )

        if (!pattern) {
            return null
        }

        const match = pattern.exec(content)

        if (!match || match.index === undefined) {
            return null
        }

        return {
            index: match.index + (match[1]?.length ?? 0),
        }
    }

    private static findSecondControlHeader(
        content: string,
        allowedTargets: readonly RunInvocationTarget[],
    ): { index: number } | null {
        const pattern = StreamingToolParser.buildControlHeaderPattern(
            allowedTargets,
            "ig",
        )

        if (!pattern) {
            return null
        }

        let seenFirstHeader = false
        let match: RegExpExecArray | null

        while ((match = pattern.exec(content)) !== null) {
            const index = match.index + (match[1]?.length ?? 0)

            if (!seenFirstHeader) {
                seenFirstHeader = true
                continue
            }

            return {
                index,
            }
        }

        return null
    }

    private getRunPrefixLookback(): number {
        const prefixes = StreamingToolParser.getControlHeaders(
            this.callbacks.allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
        )
        return Math.max(...prefixes.map((prefix) => prefix.length)) - 1
    }

    constructor(callbacks?: StreamingToolParserCallbacks) {
        if (callbacks) this.callbacks = callbacks
    }

    /** Replace callbacks (e.g. after construction). */
    on(cb: StreamingToolParserCallbacks): void {
        this.callbacks = { ...this.callbacks, ...cb }
    }

    /**
     * Whether the parser has sealed after stream completion.
     */
    get sealed(): boolean {
        return this._sealed
    }

    /** Feed a new token from the SSE stream. */
    push(token: string): void {
        if (this._sealed) return

        for (const ch of token) {
            if (this._sealed) {
                break
            }

            this.fullContent += ch

            if (this.inSegment) {
                this.segmentBuffer += ch
                this.sealAtFirstSegmentIfSecondHeaderDetected()
            } else {
                this.speechBuffer += ch
                this.captureSegmentHeaderIfPresent()
            }
        }

        if (!this.inSegment && !this._sealed) {
            this.flushSafeSpeechTail()
        }
    }

    /**
     * Signal that the stream has ended.
     *
     * If we're mid-command (truncated response), buffer it and attempt
     * recovery only after the full streamed response is available.
     */
    finish(): void {
        if (this._sealed) {
            return
        }

        if (!this.inSegment && this.speechBuffer.length > 0) {
            this.callbacks.onSpeechToken?.(this.speechBuffer)
            this.speechBuffer = ""
        }

        if (this.inSegment && this.segmentBuffer.length > 0) {
            const bufferedSegment = this.segmentBuffer.trimEnd()

            this.callbacks.onLog?.(
                "Stream ended with a structured control segment buffered for finish-time parsing",
            )
            this.inSegment = false
            this.segmentBuffer = ""

            this.tryParseFullContent(bufferedSegment)
            this._sealed = true
            return
        }

        this.tryParseFullContent()
        this._sealed = true
    }

    /** Reset internal state for reuse across turns. */
    reset(): void {
        this.speechBuffer = ""
        this.segmentBuffer = ""
        this.inSegment = false
        this._sealed = false
        this.fullContent = ""
    }

    /** Get all content accumulated so far. */
    getFullContent(): string {
        return this.fullContent
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    private captureSegmentHeaderIfPresent(): void {
        const first = StreamingToolParser.findControlHeader(
            this.speechBuffer,
            this.callbacks.allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
        )

        if (!first) {
            return
        }

        const runStart = first.index
        const speech = this.speechBuffer.slice(0, runStart)
        if (speech.length > 0) {
            this.callbacks.onSpeechToken?.(speech)
        }

        this.segmentBuffer = this.speechBuffer.slice(runStart)
        this.speechBuffer = ""
        this.inSegment = true
    }

    private flushSafeSpeechTail(): void {
        const keep = this.getRunPrefixLookback()
        if (this.speechBuffer.length <= keep) {
            return
        }

        const safeText = this.speechBuffer.slice(0, this.speechBuffer.length - keep)
        this.speechBuffer = this.speechBuffer.slice(this.speechBuffer.length - keep)
        if (safeText.length > 0) {
            this.callbacks.onSpeechToken?.(safeText)
        }
    }

    private sealAtFirstSegmentIfSecondHeaderDetected(): void {
        const secondHeader = StreamingToolParser.findSecondControlHeader(
            this.segmentBuffer,
            this.callbacks.allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
        )

        if (!secondHeader) {
            return
        }

        const contentBeforeSegment = this.fullContent.slice(
            0,
            this.fullContent.length - this.segmentBuffer.length,
        )
        const firstSegmentRaw = this.segmentBuffer.slice(0, secondHeader.index).trimEnd()
        const acceptedContent = `${contentBeforeSegment}${firstSegmentRaw}`

        this.callbacks.onLog?.(
            "Detected a second structured control header during streaming; aborting generation and keeping only the first segment",
        )

        this.fullContent = acceptedContent
        this.segmentBuffer = ""
        this.speechBuffer = ""
        this.inSegment = false

        this.tryParseContent(acceptedContent, firstSegmentRaw)
        this._sealed = true
    }

    private tryParseFullContent(rawCommandText?: string): boolean {
        return this.tryParseContent(this.fullContent, rawCommandText)
    }

    private tryParseContent(content: string, rawCommandText?: string): boolean {
        let fullResult
        try {
            fullResult = parseStructuredAgentText(content, {
                allowTruncated: true,
                allowedTargets:
                    this.callbacks.allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
            })
        } catch (error: unknown) {
            this.callbacks.onToolCallError?.(
                error instanceof Error
                    ? error
                    : new Error(String(error)),
                rawCommandText ?? content,
            )
            return true
        }

        if (!fullResult) {
            if (!this.containsStandaloneRunStart(content)) {
                return false
            }

            this.callbacks.onToolCallError?.(
                new Error(
                    "Structured command text could not be parsed after stream completion",
                ),
                rawCommandText ?? content,
            )
            return true
        }

        if (fullResult.selectedToolCall) {
            this.callbacks.onLog?.(
                `Parsed selected tool call after stream completion: ${fullResult.selectedToolCall.rawInvocation ?? fullResult.selectedToolCall.parameters.command ?? fullResult.selectedToolCall.name}`,
            )
            this.callbacks.onToolCall?.(fullResult.selectedToolCall)
        }

        return true
    }

    private containsStandaloneRunStart(content: string): boolean {
        return containsStandaloneStructuredInvocationStart(
            content,
            this.callbacks.allowedRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
        )
    }
}
