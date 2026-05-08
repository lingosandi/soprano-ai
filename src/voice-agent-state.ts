import { assign, setup } from "xstate"

import { buildStateToEventMap, createManagedActor } from "./actor-helpers"
import { AGENT_STATES, type AgentState } from "./voice-agent-types"

export type VoiceAgentMachineEvent =
    | { type: "TO_IDLE" }
    | { type: "TO_CONNECTING" }
    | { type: "TO_LISTENING" }
    | { type: "TO_THINKING" }
    | { type: "TO_SPEAKING" }
    | { type: "SET_CONTINUOUS"; value: boolean }

interface VoiceAgentContext {
    continuousMode: boolean
}

type StateTransitionEvent = Exclude<
    VoiceAgentMachineEvent,
    { type: "SET_CONTINUOUS" }
>

export const STATE_TO_EVENT = buildStateToEventMap(AGENT_STATES) as Readonly<
    Record<AgentState, StateTransitionEvent["type"]>
>

export const voiceAgentMachine = setup({
    types: {
        context: {} as VoiceAgentContext,
        events: {} as VoiceAgentMachineEvent,
    },
    actions: {
        setContinuous: assign({
            continuousMode: (_, params: { value: boolean }) => params.value,
        }),
    },
}).createMachine({
    id: "voiceAgent",
    initial: "idle",
    context: {
        continuousMode: false,
    },
    on: {
        SET_CONTINUOUS: {
            actions: {
                type: "setContinuous",
                params: ({ event }) => ({ value: event.value }),
            },
        },
    },
    states: {
        idle: {
            on: {
                TO_CONNECTING: "connecting",
                TO_LISTENING: "listening",
                TO_THINKING: "thinking",
                TO_IDLE: "idle",
            },
        },
        connecting: {
            on: {
                TO_THINKING: "thinking",
                TO_LISTENING: "listening",
                TO_IDLE: "idle",
            },
        },
        listening: {
            on: {
                TO_CONNECTING: "connecting",
                TO_THINKING: "thinking",
                TO_IDLE: "idle",
            },
        },
        thinking: {
            on: {
                TO_THINKING: "thinking",
                TO_SPEAKING: "speaking",
                TO_LISTENING: "listening",
                TO_IDLE: "idle",
            },
        },
        speaking: {
            on: {
                TO_THINKING: "thinking",
                TO_LISTENING: "listening",
                TO_IDLE: "idle",
            },
        },
    },
})

export function createVoiceAgentStateActor(
    onStateChange?: (state: AgentState) => void,
) {
    return createManagedActor(
        voiceAgentMachine,
        onStateChange
            ? (state) => onStateChange(state as AgentState)
            : undefined,
    )
}