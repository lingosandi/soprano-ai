import type { ToolRegistry } from "./tool-registry"

import type { AgentTurnRunner } from "./agent-turn-runner"
import {
    executeToolSafe,
    formatToolResultMessage,
    truncateToolResult,
} from "./tool-helpers"
import type { ToolCall, ToolResult } from "./types"

export interface ToolCallExecutionRunnerDeps {
    toolRegistry: ToolRegistry
    appendToolMessage: (content: string, toolName: string) => Promise<void>
    triggerCompactionIfNeeded?: () => Promise<void>
    onToolStart?: (
        name: string,
        parameters: Record<string, unknown>,
        index: number,
        total: number,
    ) => void
    onToolComplete?: (
        name: string,
        result: ToolResult,
        truncatedResult: string,
    ) => void
    onLog?: (message: string) => void
    onError?: (error: Error) => void
}

export interface ToolCallExecutionRunOptions {
    selectedToolCall?: ToolCall
    signal?: AbortSignal
}

export interface ToolCallExecutionResult {
    aborted: boolean
    lastToolResult?: string
}

export class ToolCallExecutionRunner
    implements AgentTurnRunner<ToolCallExecutionRunOptions, ToolCallExecutionResult>
{
    constructor(private deps: ToolCallExecutionRunnerDeps) {}

    async run(
        options: ToolCallExecutionRunOptions,
    ): Promise<ToolCallExecutionResult> {
        const { selectedToolCall, signal } = options
        let lastToolResult: string | undefined

        if (!selectedToolCall) {
            return { aborted: signal?.aborted === true, lastToolResult }
        }

        this.deps.onLog?.("Executing selected tool call…")

        if (signal?.aborted) {
            return { aborted: true, lastToolResult }
        }

        const toolCall = selectedToolCall
        this.deps.onToolStart?.(
            toolCall.name,
            toolCall.parameters,
            0,
            1,
        )

            const startedAt = Date.now()
            let result = await executeToolSafe(() =>
                this.deps.toolRegistry.executeToolCall(toolCall),
            )

            if (typeof result.duration_ms !== "number") {
                result = {
                    ...result,
                    duration_ms: Date.now() - startedAt,
                }
            }

            if (signal?.aborted) {
                return { aborted: true, lastToolResult }
            }

            if (!result.success && result.error) {
                this.deps.onError?.(new Error(result.error))
            }

            const truncatedResult = truncateToolResult(
                formatToolResultMessage(toolCall, result),
                { strategy: "head-tail" },
            )
            lastToolResult = truncatedResult

            await this.deps.appendToolMessage(truncatedResult, toolCall.name)
            this.deps.onToolComplete?.(toolCall.name, result, truncatedResult)

            this.deps.onLog?.(
                `Tool ${toolCall.name} ${result.success ? "succeeded" : "failed"}: ${this.getResultPreview(result)}`,
            )

        // No circuit breaker: always allow the model to retry after failure

        if (signal?.aborted) {
            return { aborted: true, lastToolResult }
        }

        if (this.deps.triggerCompactionIfNeeded) {
            try {
                await this.deps.triggerCompactionIfNeeded()
            } catch {
                // non-fatal
            }
        }

        return { aborted: false, lastToolResult }
    }

    private getResultPreview(result: ToolResult): string {
        const output =
            typeof result.output === "string"
                ? result.output
                : typeof result.data === "string"
                    ? result.data
                    : typeof result.error === "string"
                        ? result.error
                        : JSON.stringify(result)

        if (!output) {
            return ""
        }

        return `${output.slice(0, 200)}…`
    }
}