import type { AgentTurnRunner } from "./agent-turn-runner"
import type { ToolCall } from "./types"

export interface ToolCallingLoopTurnResult {
    continueWithoutTools?: boolean
}

export interface ToolCallingResponseLoopDeps<
    TOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
> {
    turnRunner: AgentTurnRunner<TOptions, TTurnResult>
    getSelectedToolCall: (turnResult: TTurnResult) => ToolCall | undefined
    isAborted?: (turnResult: TTurnResult) => boolean
    executeSelectedToolCall: (context: {
        selectedToolCall: ToolCall
        turnResult: TTurnResult
        options: TOptions
        iteration: number
    }) => Promise<{ aborted?: boolean } | void>
    afterToolExecution?: (context: {
        selectedToolCall: ToolCall
        turnResult: TTurnResult
        options: TOptions
        iteration: number
    }) => Promise<{ aborted?: boolean } | void>
    createContinuationOptions: (context: {
        previousOptions: TOptions
        previousTurnResult: TTurnResult
        iteration: number
    }) => Promise<TOptions> | TOptions
    maxTurns?: number
    loopLabel?: string
    onLog?: (message: string) => void
}

export class ToolCallingResponseLoop<
    TOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
>
    implements AgentTurnRunner<TOptions, TTurnResult>
{
    constructor(private deps: ToolCallingResponseLoopDeps<TOptions, TTurnResult>) {}

    async run(initialOptions: TOptions): Promise<TTurnResult> {
        const maxTurns = this.deps.maxTurns ?? 12
        let options = initialOptions

        for (let iteration = 1; iteration <= maxTurns; iteration++) {
            this.deps.onLog?.(
                `${this.loopLabel} turn ${iteration}/${maxTurns}`,
            )

            const turnResult = await this.deps.turnRunner.run(options)

            if (this.deps.isAborted?.(turnResult)) {
                return turnResult
            }

            const selectedToolCall = this.deps.getSelectedToolCall(turnResult)
            if (!selectedToolCall) {
                if (!turnResult.continueWithoutTools) {
                    return turnResult
                }

                this.deps.onLog?.(
                    `${this.loopLabel} turn ${iteration} requested a corrective retry without tool execution`,
                )

                options = await this.deps.createContinuationOptions({
                    previousOptions: options,
                    previousTurnResult: turnResult,
                    iteration,
                })

                continue
            }

            const executionResult = await this.deps.executeSelectedToolCall({
                selectedToolCall,
                turnResult,
                options,
                iteration,
            })

            if (executionResult?.aborted) {
                return turnResult
            }

            const afterExecutionResult = await this.deps.afterToolExecution?.({
                selectedToolCall,
                turnResult,
                options,
                iteration,
            })

            if (afterExecutionResult?.aborted) {
                return turnResult
            }

            options = await this.deps.createContinuationOptions({
                previousOptions: options,
                previousTurnResult: turnResult,
                iteration,
            })
        }

        throw new Error(
            `${this.loopLabel} exhausted tool-calling iterations without a final response.`,
        )
    }

    private get loopLabel(): string {
        return this.deps.loopLabel ?? "Tool-calling loop"
    }
}