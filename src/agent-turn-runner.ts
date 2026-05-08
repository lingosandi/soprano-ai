export interface AgentTurnRunner<TRunOptions, TResult> {
    run(options: TRunOptions): Promise<TResult>
}