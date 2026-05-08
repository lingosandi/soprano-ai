import type { RunInvocationTarget } from "./command-response-utils"
import { DEFAULT_ALLOWED_RUN_TARGETS } from "./run-command-registry"
import type { ToolCall, ToolDefinition, ToolResult } from "./types"

export type ProxyToolExecutor = (
    parameters: Record<string, any>,
    definition: ToolDefinition,
) => Promise<ToolResult> | ToolResult

export class ProxyTool {
    constructor(
        readonly definition: ToolDefinition,
        private readonly executor: ProxyToolExecutor,
    ) {}

    async execute(parameters: Record<string, any>): Promise<ToolResult> {
        return this.executor(parameters, this.definition)
    }
}

export interface VoiceToolRegistry {
    getToolDefinitions(): ToolDefinition[]
    executeToolCall(toolCall: ToolCall): Promise<ToolResult>
    canExecute?(): boolean
    getAvailableRunTargets?(): readonly RunInvocationTarget[]
}

export type ToolRegistry = VoiceToolRegistry

export interface CreateToolRegistryOptions {
    canExecute?: boolean
    availableRunTargets?: readonly RunInvocationTarget[]
}

export function createToolRegistry(
    tools: readonly ProxyTool[] = [],
    options: CreateToolRegistryOptions = {},
): VoiceToolRegistry {
    const toolsByName = new Map(tools.map((tool) => [tool.definition.name, tool]))

    return {
        getToolDefinitions: () => tools.map((tool) => tool.definition),
        async executeToolCall(toolCall) {
            const tool = toolsByName.get(toolCall.name)
            if (!tool) {
                return {
                    success: false,
                    error: `Tool not found: ${toolCall.name}`,
                }
            }

            return tool.execute(toolCall.parameters)
        },
        canExecute: () => options.canExecute ?? tools.length > 0,
        getAvailableRunTargets: () => options.availableRunTargets ?? DEFAULT_ALLOWED_RUN_TARGETS,
    }
}

export function createEmptyToolRegistry(): VoiceToolRegistry {
    return {
        getToolDefinitions: () => [],
        async executeToolCall(toolCall) {
            return {
                success: false,
                error: `Tool not found: ${toolCall.name}`,
            }
        },
        canExecute: () => false,
        getAvailableRunTargets: () => [],
    }
}

export function canExecuteTools(registry: VoiceToolRegistry): boolean {
    return registry.canExecute?.() ?? registry.getToolDefinitions().length > 0
}

export function getAvailableRunTargets(
    registry: VoiceToolRegistry,
): readonly RunInvocationTarget[] {
    return registry.getAvailableRunTargets?.() ?? []
}
