/**
 * Shared Agent State Machine — XState v5 formalisation of the agentic loop.
 *
 * Used by both the Hammer CLI agent and the Magic webapp agent.
 * The machine is a pure state enforcer — all async side effects (LLM
 * calls, tool execution, persistence) happen in the consuming code,
 * which sends events to drive transitions.
 *
 * State graph:
 *
 *   ┌────────┐  START   ┌──────────────┐
 *   │  idle  │ ───────▸ │  prompting   │
 *   └────────┘          └──────┬───────┘
 *                              │
 *                              ▾
 *                                                         ┌──────────────┐
 *                                                         │  analyzing   │
 *                                                         └──────┬───────┘
 *                                                                │
 *                          ┌─────────────────────────────────────┤
 *                     LLM error                             LLM_SUCCESS
 *                          │                                     │
 *                          ▾                                     ▾
 *                   ┌──────────────┐             ┌──────────────┐
 *                   │  prompting   │◂────────── │  executing   │
 *                   └──────────────┘  tools done └──────────────┘
 *                                                        │
 *                                                  TOOLS_EXECUTED
 *                                                        │
 *            ┌──────────────┐                            ▾
 *            │   updating   │◂────────────────── executing
 *            └──────┬───────┘
 *                   │ UPDATE_COMPLETE
 *                   ▾
 *            ┌──────────────┐
 *            │  prompting   │  (next iteration)
 *            └──────────────┘
 *
 *      OUTCOME_SUCCESS ──▸ done (final)
 *      OUTCOME_FAILURE ──▸ failed (final)
 *
 * @module
 */
import { setup, assign } from "xstate"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** All possible states the agent can be in. */
export type AgentMachineState =
    | "idle"
    | "prompting"
    | "analyzing"
    | "executing"
    | "updating"
    | "done"
    | "failed"

export const AGENT_MACHINE_STATES: readonly AgentMachineState[] = [
    "idle",
    "prompting",
    "analyzing",
    "executing",
    "updating",
    "done",
    "failed",
] as const

/** Truncated-tool metadata carried across iterations (Hammer-specific). */
export interface TruncatedToolInfo {
    name: string
    filePath?: string
    executionSucceeded: boolean
}

/** Machine context — all mutable state for the agent loop. */
export interface AgentMachineContext {
    /** The user's task. */
    task: string
    /** Current action counter (resumes from persisted state). */
    actionCount: number
    /** Truncated-tool metadata for continuation guidance. */
    truncatedToolInfo: TruncatedToolInfo | undefined
    /** Last tool result JSON string (for resume persistence). */
    lastToolResult: string | undefined
    /** Error message when in failed state. */
    error: string | undefined
    /** Final outcome that caused loop exit. */
    finalOutcome: string

    /** Whether ListSkills has been called this session. */
    hasCalledListSkills: boolean
}

/** Events the machine accepts. */
export type AgentMachineEvent =
    | { type: "START"; task: string; actionCount?: number }
    | { type: "PROMPT_COMPLETE" }
    | { type: "LLM_SUCCESS" }
    | { type: "LLM_VALIDATION_ERROR"; error: string }
    | { type: "LLM_API_ERROR"; error: string }
    | { type: "OUTCOME_SUCCESS" }
    | { type: "OUTCOME_FAILURE"; error?: string }
    | { type: "TOOLS_EXECUTED"; lastToolResult?: string }
    | { type: "ENFORCEMENT_BREAK"; lastToolResult: string }
    | { type: "NO_TOOLS" }
    | { type: "UPDATE_COMPLETE" }
    | { type: "SET_TRUNCATION"; info: TruncatedToolInfo | undefined }
    | { type: "RESTORE_CONTEXT"; context: Partial<AgentMachineContext> }

// ---------------------------------------------------------------------------
// Machine definition
// ---------------------------------------------------------------------------

export const agentMachine = setup({
    types: {
        context: {} as AgentMachineContext,
        events: {} as AgentMachineEvent,
    },
    guards: {},
    actions: {
        assignTask: assign(({ context }, params: { task: string; actionCount?: number }) => ({
            ...context,
            task: params.task,
            actionCount: params.actionCount ?? 0,
            error: undefined,
            finalOutcome: "running",
        })),
        assignLlmError: assign({
            error: (_, params: { error: string }) => params.error,
        }),
        assignOutcomeFailure: assign(({ context }, params: { error?: string }) => ({
            ...context,
            finalOutcome: "failure",
            error: params.error,
        })),
        assignOutcomeSuccess: assign({
            finalOutcome: () => "success",
        }),
        assignToolsExecuted: assign(({ context }, params: { lastToolResult?: string }) => ({
            ...context,
            lastToolResult: params.lastToolResult,
        })),
        assignEnforcementBreak: assign({
            lastToolResult: (_, params: { lastToolResult: string }) =>
                params.lastToolResult,
        }),
        incrementAction: assign({
            actionCount: ({ context }) => context.actionCount + 1,
        }),
        setTruncation: assign({
            truncatedToolInfo: (_, params: { info: TruncatedToolInfo | undefined }) =>
                params.info,
        }),
        restoreContext: assign(
            ({ context }, params: { context: Partial<AgentMachineContext> }) => ({
                ...context,
                ...params.context,
            }),
        ),
    },
}).createMachine({
    id: "agent",
    initial: "idle",
    context: {
        task: "",
        actionCount: 0,
        truncatedToolInfo: undefined,
        lastToolResult: undefined,
        error: undefined,
        finalOutcome: "running",
        hasCalledListSkills: false,
    },

    // Global events that can be sent from any state
    on: {
        SET_TRUNCATION: {
            actions: {
                type: "setTruncation",
                params: ({ event }) => ({ info: event.info }),
            },
        },
        RESTORE_CONTEXT: {
            actions: {
                type: "restoreContext",
                params: ({ event }) => ({ context: event.context }),
            },
        },
    },

    states: {
        idle: {
            on: {
                START: {
                    target: "prompting",
                    actions: {
                        type: "assignTask",
                        params: ({ event }) => ({
                            task: event.task,
                            actionCount: event.actionCount,
                        }),
                    },
                },
                OUTCOME_FAILURE: {
                    target: "failed",
                    actions: {
                        type: "assignOutcomeFailure",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
            },
        },

        prompting: {
            on: {
                PROMPT_COMPLETE: {
                    target: "analyzing",
                    actions: { type: "incrementAction" },
                },
                OUTCOME_FAILURE: {
                    target: "failed",
                    actions: {
                        type: "assignOutcomeFailure",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
            },
        },

        analyzing: {
            on: {
                LLM_SUCCESS: {
                    target: "executing",
                },
                LLM_VALIDATION_ERROR: {
                    target: "prompting",
                    actions: {
                        type: "assignLlmError",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
                LLM_API_ERROR: {
                    target: "prompting",
                    actions: {
                        type: "assignLlmError",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
                OUTCOME_SUCCESS: {
                    target: "done",
                    actions: { type: "assignOutcomeSuccess" },
                },
                OUTCOME_FAILURE: {
                    target: "failed",
                    actions: {
                        type: "assignOutcomeFailure",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
            },
        },

        executing: {
            on: {
                TOOLS_EXECUTED: {
                    target: "updating",
                    actions: {
                        type: "assignToolsExecuted",
                        params: ({ event }) => ({
                            lastToolResult: event.lastToolResult,
                        }),
                    },
                },
                ENFORCEMENT_BREAK: {
                    target: "prompting",
                    actions: {
                        type: "assignEnforcementBreak",
                        params: ({ event }) => ({
                            lastToolResult: event.lastToolResult,
                        }),
                    },
                },
                NO_TOOLS: {
                    target: "prompting",
                },
                OUTCOME_SUCCESS: {
                    target: "done",
                    actions: { type: "assignOutcomeSuccess" },
                },
                OUTCOME_FAILURE: {
                    target: "failed",
                    actions: {
                        type: "assignOutcomeFailure",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
            },
        },

        updating: {
            on: {
                UPDATE_COMPLETE: "prompting",
                OUTCOME_FAILURE: {
                    target: "failed",
                    actions: {
                        type: "assignOutcomeFailure",
                        params: ({ event }) => ({ error: event.error }),
                    },
                },
                OUTCOME_SUCCESS: {
                    target: "done",
                    actions: { type: "assignOutcomeSuccess" },
                },
            },
        },

        done: {
            type: "final",
        },

        failed: {
            type: "final",
        },
    },
})
