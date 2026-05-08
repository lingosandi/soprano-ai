import type { AgentTurnRunner } from "./agent-turn-runner"
import {
    ToolCallingResponseLoop,
    type ToolCallingLoopTurnResult,
} from "./tool-calling-response-loop"
import type { ToolCall } from "./types"

export interface AgentTurnBuildOptionsContext<
    TRunOptions,
    TLoopOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
> {
    runOptions: TRunOptions
    iteration: number
    isContinuation: boolean
    previousOptions?: TLoopOptions
    previousTurnResult?: TTurnResult
}

export interface AgentTurnToolContext<
    TRunOptions,
    TLoopOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
> {
    runOptions: TRunOptions
    loopOptions: TLoopOptions
    turnResult: TTurnResult
    selectedToolCall: ToolCall
    iteration: number
}

interface AgentTurnFinalizeContext<
    TRunOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
> {
    runOptions: TRunOptions
    finalTurnResult: TTurnResult
}

interface AgentTurnOrchestratorBaseDeps<
    TRunOptions,
    TLoopOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
> {
    turnRunner: AgentTurnRunner<TLoopOptions, TTurnResult>
    getSelectedToolCall: (turnResult: TTurnResult) => ToolCall | undefined
    isAborted?: (context: {
        runOptions: TRunOptions
        turnResult: TTurnResult
    }) => boolean
    executeSelectedToolCall: (
        context: AgentTurnToolContext<TRunOptions, TLoopOptions, TTurnResult>,
    ) => Promise<{ aborted?: boolean } | void>
    afterToolExecution?: (
        context: AgentTurnToolContext<TRunOptions, TLoopOptions, TTurnResult>,
    ) => Promise<{ aborted?: boolean } | void>
    beforeTurn?: (
        context: AgentTurnBuildOptionsContext<TRunOptions, TLoopOptions, TTurnResult>,
    ) => Promise<void> | void
    createTurnOptions: (
        context: AgentTurnBuildOptionsContext<TRunOptions, TLoopOptions, TTurnResult>,
    ) => Promise<TLoopOptions> | TLoopOptions
    maxTurns?: number
    loopLabel?: string
    onLog?: (message: string) => void
}

export type AgentTurnOrchestratorDeps<
    TRunOptions,
    TLoopOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
    TResult = TTurnResult,
> = AgentTurnOrchestratorBaseDeps<
    TRunOptions,
    TLoopOptions,
    TTurnResult
> & (
    [TResult] extends [TTurnResult]
        ? {
            finalizeResult?: (
                context: AgentTurnFinalizeContext<TRunOptions, TTurnResult>,
            ) => Promise<TResult> | TResult
        }
        : {
            finalizeResult: (
                context: AgentTurnFinalizeContext<TRunOptions, TTurnResult>,
            ) => Promise<TResult> | TResult
        }
)

/**
 * Higher-level shared turn orchestrator that prepares each turn, runs the
 * shared tool-calling loop, and applies transport-specific finalization.
 */
export class AgentTurnOrchestrator<
    TRunOptions,
    TLoopOptions,
    TTurnResult extends ToolCallingLoopTurnResult,
    TResult = TTurnResult,
> implements AgentTurnRunner<TRunOptions, TResult>
{
    constructor(
        private deps: AgentTurnOrchestratorDeps<
            TRunOptions,
            TLoopOptions,
            TTurnResult,
            TResult
        >,
    ) {}

    async run(runOptions: TRunOptions): Promise<TResult> {
        const buildTurnOptions = async (
            iteration: number,
            isContinuation: boolean,
            previousOptions?: TLoopOptions,
            previousTurnResult?: TTurnResult,
        ): Promise<TLoopOptions> => {
            const context = {
                runOptions,
                iteration,
                isContinuation,
                previousOptions,
                previousTurnResult,
            }

            await this.deps.beforeTurn?.(context)

            return this.deps.createTurnOptions(context)
        }

        const responseLoop = new ToolCallingResponseLoop<TLoopOptions, TTurnResult>({
            turnRunner: this.deps.turnRunner,
            getSelectedToolCall: this.deps.getSelectedToolCall,
            isAborted: this.deps.isAborted
                ? (turnResult) =>
                    this.deps.isAborted?.({
                        runOptions,
                        turnResult,
                    }) ?? false
                : undefined,
            executeSelectedToolCall: async ({
                selectedToolCall,
                turnResult,
                options,
                iteration,
            }) =>
                this.deps.executeSelectedToolCall({
                    runOptions,
                    loopOptions: options,
                    turnResult,
                    selectedToolCall,
                    iteration,
                }),
            afterToolExecution: this.deps.afterToolExecution
                ? async ({
                    selectedToolCall,
                    turnResult,
                    options,
                    iteration,
                }) =>
                    this.deps.afterToolExecution?.({
                        runOptions,
                        loopOptions: options,
                        turnResult,
                        selectedToolCall,
                        iteration,
                    })
                : undefined,
            createContinuationOptions: ({
                previousOptions,
                previousTurnResult,
                iteration,
            }) =>
                buildTurnOptions(
                    iteration + 1,
                    true,
                    previousOptions,
                    previousTurnResult,
                ),
            maxTurns: this.deps.maxTurns,
            loopLabel: this.deps.loopLabel,
            onLog: this.deps.onLog,
        })

        const initialOptions = await buildTurnOptions(1, false)
        const finalTurnResult = await responseLoop.run(initialOptions)

        return this.finalizeTurnResult(runOptions, finalTurnResult)
    }

    private finalizeTurnResult(
        runOptions: TRunOptions,
        finalTurnResult: TTurnResult,
    ): TResult | Promise<TResult> {
        if (this.deps.finalizeResult) {
            return this.deps.finalizeResult({
                runOptions,
                finalTurnResult,
            })
        }

        return finalTurnResult as unknown as TResult
    }
}