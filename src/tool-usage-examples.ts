import type {
    ToolDefinition,
    ToolDefinitionMetadata,
    ToolParameterDefinition,
} from "./types"

function isBooleanType(type: string | string[]): boolean {
    return (Array.isArray(type) ? type : [type]).includes("boolean")
}

function isNumberType(type: string | string[]): boolean {
    return (Array.isArray(type) ? type : [type]).some((entry) =>
        entry === "number" || entry === "integer",
    )
}

function isStringType(type: string | string[]): boolean {
    return (Array.isArray(type) ? type : [type]).includes("string")
}

function toFlagName(name: string): string {
    return name.replace(/_/g, "-")
}

function derivePositionalParams(
    parameters: Record<string, ToolParameterDefinition>,
): string[] {
    return Object.entries(parameters)
        .filter(([, definition]) => definition.required && !isBooleanType(definition.type) && definition.positional !== false)
        .map(([name]) => name)
}

function derivePassthroughParam(
    metadata: ToolDefinitionMetadata | undefined,
    parameters: Record<string, ToolParameterDefinition>,
): string | undefined {
    if (!metadata?.capabilities?.includes("raw_command_args")) {
        return undefined
    }

    const entries = Object.entries(parameters)
    if (entries.length !== 1) {
        return undefined
    }

    const [parameterName, definition] = entries[0]
    if (!definition.required || !isStringType(definition.type)) {
        return undefined
    }

    return parameterName
}

function quoteIfNeeded(value: string): string {
    if (/^[A-Za-z0-9_./:@?&=%+-]+$/.test(value)) {
        return value
    }

    return JSON.stringify(value)
}

function buildStringSample(parameterName: string): string {
    const normalized = parameterName.toLowerCase()

    if (normalized.includes("query")) return "bitcoin price today"
    if (normalized.includes("url")) return "https://example.com"
    if (normalized === "path" || normalized.endsWith("_path") || normalized.includes("file")) return "README.md"
    if (normalized.includes("skill")) return "frontend-design"
    if (normalized.includes("command") || normalized === "args") return "cat README.md"
    if (normalized.includes("action")) return "start"
    if (normalized.includes("filename")) return "session-note"
    if (normalized.includes("board")) return "board-1"
    if (normalized.includes("element")) return "element-1"
    if (normalized.includes("id")) return "item-1"
    if (normalized.includes("prompt")) return "Generate a hero image"
    if (normalized.includes("text") || normalized.includes("message")) return "example text"

    return "example"
}

function buildSampleValue(
    parameterName: string,
    definition: ToolParameterDefinition,
): string {
    if (definition.enum && definition.enum.length > 0) {
        const first = definition.enum[0]
        return typeof first === "string" ? quoteIfNeeded(first) : JSON.stringify(first)
    }

    if (isBooleanType(definition.type)) {
        return "true"
    }

    if (isNumberType(definition.type)) {
        return "1"
    }

    const normalizedTypes = Array.isArray(definition.type) ? definition.type : [definition.type]
    if (normalizedTypes.includes("array")) {
        return '["example"]'
    }
    if (normalizedTypes.includes("object")) {
        return '{"key":"value"}'
    }
    if (normalizedTypes.includes("null")) {
        return "null"
    }

    return quoteIfNeeded(buildStringSample(parameterName))
}

export function buildToolUsageExample(
    tool: Pick<ToolDefinition, "name" | "parameters" | "metadata">,
): string {
    const passthroughParam = derivePassthroughParam(tool.metadata, tool.parameters)
    if (passthroughParam) {
        return `${tool.name} ${buildSampleValue(passthroughParam, tool.parameters[passthroughParam])}`
    }

    const parts = [tool.name]
    const positionalParams = derivePositionalParams(tool.parameters)

    for (const parameterName of positionalParams) {
        parts.push(buildSampleValue(parameterName, tool.parameters[parameterName]))
    }

    const requiredFlags = Object.entries(tool.parameters)
        .filter(([name, definition]) => !positionalParams.includes(name) && definition.required)

    for (const [parameterName, definition] of requiredFlags) {
        const flag = `--${toFlagName(parameterName)}`
        if (isBooleanType(definition.type)) {
            parts.push(flag)
            continue
        }

        parts.push(flag, buildSampleValue(parameterName, definition))
    }

    return parts.join(" ")
}