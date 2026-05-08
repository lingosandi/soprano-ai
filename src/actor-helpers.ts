/**
 * Shared xstate actor utilities — reusable infrastructure for creating
 * and managing xstate v5 actors across the mono-repo.
 *
 * Used by:
 *  - Voice agent state machine (packages./voice-agent-service.ts)
 *  - Hammer CLI agent state machine (apps/hammer/src/agents/agentMachine.ts)
 *
 * @module
 */
import { createActor, type AnyStateMachine, type Actor } from "xstate"

/**
 * Creates an xstate actor with a standard subscription that deduplicates
 * state-value changes (ignoring context-only updates) and calls back
 * only when the state name itself changes.
 *
 * This pattern is shared between the voice agent and hammer agent.
 *
 * @returns The started actor and a cleanup function.
 */
export function createManagedActor<TMachine extends AnyStateMachine>(
    machine: TMachine,
    onStateChange?: (state: string) => void,
): {
    actor: Actor<TMachine>
    stop: () => void
} {
    const actor = createActor(machine)

    let lastEmittedState: string | null = null
    actor.subscribe((snapshot: any) => {
        if (snapshot.status !== "active") return
        const currentState = snapshot.value as string
        if (currentState === lastEmittedState) return
        lastEmittedState = currentState
        onStateChange?.(currentState)
    })

    actor.start()

    return {
        actor,
        stop: () => actor.stop(),
    }
}

/**
 * Helper to read the current state value from a running actor.
 */
export function getActorState<TMachine extends AnyStateMachine>(
    actor: Actor<TMachine>,
): string {
    return (actor.getSnapshot() as any).value as string
}

/**
 * Helper to read context from a running actor.
 */
export function getActorContext<TMachine extends AnyStateMachine>(
    actor: Actor<TMachine>,
): any {
    return (actor.getSnapshot() as any).context
}

/**
 * Builds a `STATE_TO_EVENT` mapping from a list of state names.
 *
 * Convention: event type is `TO_<UPPER_SNAKE_CASE>` of the state name.
 * e.g. "idle" → "TO_IDLE", "executingTools" → "TO_EXECUTING_TOOLS"
 *
 * Used to bridge imperative `setState("idle")` calls with event-driven
 * machine dispatch.
 */
export function buildStateToEventMap<S extends string>(
    states: readonly S[],
): Readonly<Record<S, string>> {
    const map: Record<string, string> = {}
    for (const state of states) {
        // camelCase → UPPER_SNAKE_CASE
        const upper = state
            .replace(/([A-Z])/g, "_$1")
            .toUpperCase()
        map[state] = `TO_${upper}`
    }
    return map as Readonly<Record<S, string>>
}
