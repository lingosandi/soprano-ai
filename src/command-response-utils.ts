import type {
    LoopOutcome,
    ToolCall,
    ToolDefinition,
    ToolParameterDefinition,
} from "./types"
import {
    DEFAULT_ALLOWED_RUN_TARGETS,
    DEFAULT_RUN_COMMAND_REGISTRY,
    SUPPORTED_RUN_TARGETS,
    type RunCommandParseResult,
    type RunCommandRegistry,
    type RunInvocationTarget,
} from "./run-command-registry"
import { buildMultipleStructuredControlSegmentsValidationError } from "./tool-call-prompts"
import {
    formatUnixToolSurface,
    parseUnixToolCommand,
} from "./unix-tooling"

export {
    DEFAULT_ALLOWED_RUN_TARGETS,
}
export type {
    RunInvocationTarget,
}

function formatParameterType(type: string | string[]): string {
    if (Array.isArray(type)) {
        return type.join("|")
    }
    return type
}

export function formatToolDefinitions(
    tools: ToolDefinition[],
    style: "compact" | "verbose" | "unix" = "compact",
): string {
    if (style === "unix") {
        return formatUnixToolSurface(tools)
    }

    if (style === "verbose") {
        return tools
            .map((tool) => {
                const params = Object.entries(tool.parameters)
                    .map(
                        ([name, definition]) =>
                            `    ${name} (${formatParameterType(definition.type)}${definition.required ? ", required" : ""}): ${definition.description}`,
                    )
                    .join("\n")
                return `- ${tool.name}: ${tool.description}\n  Parameters:\n${params}`
            })
            .join("\n\n")
    }

    return tools
        .map((tool) => {
            const params = Object.entries(tool.parameters)
                .map(([name, schema]) => {
                    const required = schema.required ? "*" : ""
                    return `${name}${required}: ${formatParameterType(schema.type)}`
                })
                .join(", ")
            return `• ${tool.name}(${params}) - ${tool.description}`
        })
        .join("\n")
}

function normalizeExpectedTypes(type: string | string[]): string[] {
    return (Array.isArray(type) ? type : [type]).map((entry) =>
        entry.toLowerCase(),
    )
}

function parseStructuredLiteral(rawValue: string): unknown {
    try {
        return JSON.parse(rawValue)
    } catch {
        return undefined
    }
}

function coerceArrayFallbackItem(
    rawValue: string,
    definition: ToolParameterDefinition,
): unknown {
    if (!definition.items?.type) {
        return rawValue
    }

    return coerceStringParameterValue(rawValue, {
        type: definition.items.type,
        description: definition.items.description ?? "",
        items: definition.items.items,
        properties: definition.items.properties,
        additionalProperties: definition.items.additionalProperties,
        default: definition.items.default,
    })
}

function coerceStringParameterValue(
    rawValue: string,
    definition: ToolParameterDefinition,
): unknown {
    const trimmed = rawValue.trim()
    if (trimmed.length === 0) return rawValue

    for (const expectedType of normalizeExpectedTypes(definition.type)) {
        switch (expectedType) {
            case "array": {
                const parsed = parseStructuredLiteral(trimmed)
                if (Array.isArray(parsed)) return parsed

                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                    const itemTypes = definition.items?.type
                        ? normalizeExpectedTypes(definition.items.type)
                        : []
                    if (itemTypes.includes("object")) {
                        return [parsed]
                    }
                }

                if (!/[{}\[\]]/.test(trimmed)) {
                    const parts = trimmed
                        .split(/[\r\n,]+/)
                        .map((part) => part.trim())
                        .filter(Boolean)
                    if (parts.length > 0) {
                        return parts.map((part) =>
                            coerceArrayFallbackItem(part, definition),
                        )
                    }
                }
                break
            }

            case "object": {
                const parsed = parseStructuredLiteral(trimmed)
                if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                    return parsed
                }
                break
            }

            case "boolean": {
                if (trimmed === "true") return true
                if (trimmed === "false") return false
                break
            }

            case "integer": {
                if (/^-?(?:0|[1-9]\d*)$/.test(trimmed)) {
                    return Number(trimmed)
                }
                break
            }

            case "number": {
                if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$/.test(trimmed)) {
                    return Number(trimmed)
                }
                break
            }

            case "null": {
                if (trimmed === "null") return null
                break
            }
        }
    }

    return rawValue
}

export function coerceToolCallToDefinition(
    call: ToolCall | undefined,
    tools: ToolDefinition[] = [],
): ToolCall | undefined {
    if (!call || tools.length === 0) return call

    const definitionsByName = new Map(tools.map((tool) => [tool.name, tool]))
    const definition = definitionsByName.get(call.name)

    if (
        definition &&
        call.kind !== "bash" &&
        typeof call.rawInvocation === "string" &&
        call.rawInvocation.trim().length > 0
    ) {
        const parsed = parseUnixToolCommand(definition, call.rawInvocation, {
            allowTruncated: call.truncated === true,
        })
        if (parsed.ok) {
            return {
                ...call,
                parameters: parsed.parameters as Record<string, any>,
            }
        }
    }

    if (!definition) return call

    let didCoerce = false
    const parameters = { ...call.parameters }

    for (const [paramName, paramValue] of Object.entries(call.parameters)) {
        const paramDefinition = definition.parameters[paramName]
        if (!paramDefinition || typeof paramValue !== "string") continue

        const coercedValue = coerceStringParameterValue(
            paramValue,
            paramDefinition,
        )
        if (coercedValue !== paramValue) {
            parameters[paramName] = coercedValue
            didCoerce = true
        }
    }

    return didCoerce
        ? {
            ...call,
            parameters,
        }
        : call
}

export interface ExtractedRunInvocation {
    target: "tool" | "bash" | "background_bash"
    command: string
    raw: string
    start: number
    end: number
    truncated: boolean
}

interface ExtractedRunInvocationPayload {
    command: string
    end: number
    truncated: boolean
    quoted: boolean
}

export interface ParsedStructuredAgentText {
    prose: string
    outcome?: LoopOutcome
    selectedToolCall?: ToolCall
    selectedToolCallCount: number
}

function buildStructuredInvocationHeaderPattern(
    allowedTargets: readonly RunInvocationTarget[],
): RegExp {
    const normalizedTargets = Array.from(new Set(allowedTargets)).filter(
        (target): target is RunInvocationTarget => SUPPORTED_RUN_TARGETS.includes(target),
    )
    if (normalizedTargets.length === 0) {
        return /$^/
    }

    return new RegExp(
        `(^|\\n)([ \t]*)---(${normalizedTargets.join("|")})---(?:[ \t]*\\r?\\n[ \t]*|[ \t]+|$)`,
        "gi",
    )
}

function extractInlineTrailingCompatibilityInvocation(
    content: string,
    options?: {
        allowTruncated?: boolean
        allowedTargets?: readonly RunInvocationTarget[]
    },
): ExtractedRunInvocation | null {
    const allowTruncated = options?.allowTruncated === true
    const allowedTargets = options?.allowedTargets ?? DEFAULT_ALLOWED_RUN_TARGETS
    const normalizedTargets = Array.from(new Set(allowedTargets)).filter(
        (target): target is RunInvocationTarget => SUPPORTED_RUN_TARGETS.includes(target),
    )

    if (normalizedTargets.length === 0) {
        return null
    }

    let lastHeaderMatch: RegExpExecArray | null = null
    const headerMatcher = new RegExp(`---(${normalizedTargets.join("|")})---`, "gi")
    let match: RegExpExecArray | null

    while ((match = headerMatcher.exec(content)) !== null) {
        lastHeaderMatch = match
    }

    const matchedTarget = lastHeaderMatch?.[1]?.toLowerCase()
    if (!lastHeaderMatch?.[0] || !matchedTarget) {
        return null
    }

    const prefix = content.slice(0, lastHeaderMatch.index).trimEnd()
    if (!/[.!?]$/.test(prefix)) {
        return null
    }

    const payloadSource = content.slice(lastHeaderMatch.index + lastHeaderMatch[0].length)
    const payloadMatch = payloadSource.match(/^[ \t]+([^\r\n][^\r\n]*?)\s*$/)
    const normalizedCommand = payloadMatch?.[1]?.trim()

    if (!normalizedCommand) {
        return null
    }

    const target = matchedTarget === "tool"
        ? "tool"
        : matchedTarget === "background_bash"
            ? "background_bash"
            : "bash"

    return {
        target,
        command: normalizedCommand,
        raw: content.slice(lastHeaderMatch.index).trimEnd(),
        start: lastHeaderMatch.index,
        end: content.length,
        truncated: allowTruncated
            && isLikelyTruncatedShellLikePayload(normalizedCommand),
    }
}

export function containsStandaloneStructuredInvocationStart(
    content: string,
    allowedTargets: readonly RunInvocationTarget[] = DEFAULT_ALLOWED_RUN_TARGETS,
): boolean {
    const matcher = buildStructuredInvocationHeaderPattern(allowedTargets)
    return matcher.test(content)
}

export function extractRunInvocations(
    content: string,
    options?: {
        allowTruncated?: boolean
        allowedTargets?: readonly RunInvocationTarget[]
    },
): ExtractedRunInvocation[] {
    const invocations: ExtractedRunInvocation[] = []
    const allowTruncated = options?.allowTruncated === true
    const allowedTargets = options?.allowedTargets ?? DEFAULT_ALLOWED_RUN_TARGETS
    const matcher = buildStructuredInvocationHeaderPattern(allowedTargets)
    const headers: Array<{
        target: ExtractedRunInvocation["target"]
        start: number
        payloadStart: number
    }> = []
    let match: RegExpExecArray | null

    while ((match = matcher.exec(content)) !== null) {
        const matchedTarget = match[3]?.toLowerCase()
        const target = matchedTarget === "tool"
            ? "tool"
            : matchedTarget === "background_bash"
                ? "background_bash"
                : "bash"
        const prefixLength = (match[1] ?? "").length
        const start = match.index + prefixLength
        const payloadStart = match.index + match[0].length

        headers.push({
            target,
            start,
            payloadStart,
        })
    }

    if (headers.length === 0) {
        const inlineCompatibilityInvocation = extractInlineTrailingCompatibilityInvocation(
            content,
            options,
        )

        if (inlineCompatibilityInvocation) {
            return [inlineCompatibilityInvocation]
        }
    }

    for (const [index, header] of headers.entries()) {
        const end = headers[index + 1]?.start ?? content.length
        const raw = content.slice(header.start, end).trimEnd()
        const command = content.slice(header.payloadStart, end).trimEnd()
        const normalizedCommand = command.trim()

        if (!normalizedCommand) {
            continue
        }

        invocations.push({
            target: header.target,
            command: normalizedCommand,
            raw,
            start: header.start,
            end,
            truncated: allowTruncated
                && end === content.length
                && isLikelyTruncatedShellLikePayload(normalizedCommand),
        })
    }

    return invocations
}

export function parseStructuredAgentText(
    content: string,
    options?: {
        allowTruncated?: boolean
        allowedTargets?: readonly RunInvocationTarget[]
        commandRegistry?: RunCommandRegistry
    },
): ParsedStructuredAgentText | null {
    const invocations = extractRunInvocations(content, options)
    if (invocations.length === 0) {
        return null
    }

    if (invocations.length > 1) {
        throw new Error(
            buildMultipleStructuredControlSegmentsValidationError(invocations.length),
        )
    }

    let selectedToolCall: ToolCall | undefined
    let selectedToolCallCount = 0
    let outcome: ParsedStructuredAgentText["outcome"]
    let latestSuccessfulInvocationIndex = -1
    let latestInvocationError: unknown
    let latestInvocationErrorIndex = -1
    const commandRegistry = options?.commandRegistry ?? DEFAULT_RUN_COMMAND_REGISTRY

    for (const [index, invocation] of invocations.entries()) {
        let parsedInvocation: RunCommandParseResult | null
        try {
            parsedInvocation = commandRegistry.parseInvocation(invocation, {
                allowTruncated: options?.allowTruncated === true,
            })
        } catch (error: unknown) {
            latestInvocationError = error
            latestInvocationErrorIndex = index
            continue
        }

        if (!parsedInvocation) {
            continue
        }

        latestSuccessfulInvocationIndex = index

        if (parsedInvocation.selectedToolCall) {
            selectedToolCall = parsedInvocation.selectedToolCall
        }

        if (
            parsedInvocation.outcome
            && parsedInvocation.outcome !== "continue"
            && !parsedInvocation.selectedToolCall
        ) {
            selectedToolCall = undefined
        }

        selectedToolCallCount += parsedInvocation.selectedToolCallCount ?? 0

        if (parsedInvocation.outcome) {
            outcome = parsedInvocation.outcome
        }
    }

    if (latestInvocationErrorIndex > latestSuccessfulInvocationIndex) {
        throw latestInvocationError
    }

    if (!outcome && !selectedToolCall) {
        if (latestInvocationError !== undefined) {
            throw latestInvocationError
        }

        return null
    }

    return {
        prose: stripInvocations(content, invocations),
        outcome,
        ...(selectedToolCall ? { selectedToolCall } : {}),
        selectedToolCallCount,
    }
}

function isLikelyTruncatedShellLikePayload(command: string): boolean {
    let quote: '"' | "'" | "`" | null = null
    let escaping = false

    for (const char of command) {
        if (quote) {
            if (escaping) {
                escaping = false
                continue
            }

            if (quote !== "'" && char === "\\") {
                escaping = true
                continue
            }

            if (char === quote) {
                quote = null
            }

            continue
        }

        if (escaping) {
            escaping = false
            continue
        }

        if (char === "\\") {
            escaping = true
            continue
        }

        if (char === '"' || char === "'" || char === "`") {
            quote = char
        }
    }

    if (quote || escaping) {
        return true
    }

    const heredocMatches = Array.from(command.matchAll(/<<-?\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1/g))
    if (heredocMatches.length === 0) {
        return false
    }

    const lines = command.split(/\r?\n/)

    return heredocMatches.some((match) => {
        const terminator = match[2]
        if (!terminator) {
            return false
        }

        return !lines.some((line) => line.trim() === terminator)
    })
}

function stripInvocations(
    content: string,
    invocations: ExtractedRunInvocation[],
): string {
    if (invocations.length === 0) {
        return content.trim()
    }

    let cursor = 0
    let prose = ""

    for (const invocation of invocations) {
        prose += content.slice(cursor, invocation.start)
        cursor = invocation.end
    }

    prose += content.slice(cursor)
    return prose
        .split(/\r?\n/)
        .map((line) => line.trimEnd())
        .join("\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim()
}

