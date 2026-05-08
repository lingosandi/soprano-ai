import { createToolRegistry, ProxyTool, type ToolRegistry } from "../../src/tool-registry"
import { parseUnixToolCommand } from "../../src/unix-tooling"
import type { ToolCall, ToolDefinition, ToolResult } from "../../src/types"

export function createTestToolRegistry(
    tools: ToolDefinition[] = [],
    toolExecutor?: (toolCall: ToolCall) => Promise<ToolResult>,
): ToolRegistry | undefined {
    if (tools.length === 0) {
        return undefined
    }

    const toolDefinitions = new Map(tools.map((definition) => [definition.name, definition]))

    const executor = toolExecutor
        ? async (toolCall: ToolCall & { command?: string }) => {
            if (typeof toolCall.command === "string") {
                const definition = toolDefinitions.get(toolCall.name)
                if (definition) {
                    const parsed = parseUnixToolCommand(definition, toolCall.command)
                    if (parsed.ok) {
                        return toolExecutor({
                            name: toolCall.name,
                            parameters: parsed.parameters as Record<string, any>,
                        })
                    }
                }
            }

            return toolExecutor({
                name: toolCall.name,
                parameters: toolCall.parameters,
            })
        }
        : async (toolCall: ToolCall) => {
            throw new Error(`No tool executor is available for ${toolCall.name}.`)
        }

    return createToolRegistry(
        tools.map((definition) =>
            new ProxyTool(
                definition,
                (parameters, proxiedDefinition) => executor({
                    name: proxiedDefinition.name,
                    parameters,
                }),
            )
        ),
        {
            canExecute: toolExecutor ? undefined : false,
        },
    )
}