import type {
    ToolCall,
    ToolDefinition,
    ToolParameterDefinition,
    ToolResult,
} from "./types"
import { buildToolUsageExample } from "./tool-usage-examples"
import {
    UNIX_PASSTHROUGH_TOOL_GUIDANCE_LINE,
    UNIX_TOOL_USAGE_GUIDANCE_LINE,
} from "./tool-call-prompts"

type ChainOperator = "|" | "&&" | "||" | ";"

export interface CommandRuntime {
    getToolDefinitions(): ToolDefinition[]
    executeTool(name: string, parameters: Record<string, any>): Promise<ToolResult>
    executeBash?(command: string): Promise<ToolResult>
    executeBackgroundBash?(command: string): Promise<ToolResult>
}

export interface CommandTargetInfo {
    name: string
    path?: string
    command?: string
}

interface CommandDescriptor {
    command: string
    summary: string
    details?: string
    usage: string
    example: string
    toolName: string
    parameters: Record<string, ToolParameterDefinition>
    positionalParams: string[]
    passthroughParam?: string
}

interface ParsedCommandArgs {
    positional: string[]
    flags: Record<string, unknown>
    unknownFlags: string[]
    error?: string
}

export function isBashToolCall(toolCall: Pick<ToolCall, "name" | "kind">): boolean {
    return toolCall.kind === "bash"
}

export function isBackgroundBashToolCall(toolCall: Pick<ToolCall, "name" | "kind">): boolean {
    return toolCall.kind === "background_bash"
}

export function resolveToolDefinitionForInvocation(
    tools: ToolDefinition[],
    invocationName: string,
): ToolDefinition | undefined {
    return tools.find((tool) => tool.name === invocationName)
}

export function formatUnixToolSurface(tools: ToolDefinition[]): string {
    if (tools.length === 0) {
        return ""
    }

    const descriptors = tools
        .map((tool) => buildToolDescriptor(tool))
        .sort((left, right) => left.command.localeCompare(right.command))

    return [
        '## Registered Tools',
        '',
        '- Invoke them with a standalone `---tool---` header followed by the tool payload on the next line.',
        `- ${UNIX_TOOL_USAGE_GUIDANCE_LINE}`,
        `- ${UNIX_PASSTHROUGH_TOOL_GUIDANCE_LINE}`,
        ...descriptors.flatMap((descriptor) => {
            const line = `### \`${descriptor.usage}\``
            const details: string[] = []

            details.push(`- ${descriptor.summary}`)

            if (descriptor.details) {
                details.push(
                    descriptor.details
                        .split("\n")
                        .map((l) => `- ${l}`)
                        .join("\n"),
                )
            }

            details.push(`- Example: \`${descriptor.example}\``)

            return [line, ...details, ""]
        }),
    ].join("\n").trim()
}

export function formatToolCallAsUnixCommand(
    toolCall: Pick<ToolCall, "name" | "kind" | "parameters" | "rawInvocation">,
    toolDefinitions: ToolDefinition[] = [],
): string | undefined {
    if (isBashToolCall(toolCall) || isBackgroundBashToolCall(toolCall)) {
        const command = typeof toolCall.parameters?.command === "string"
            ? toolCall.parameters.command.trim()
            : ""
        return command || undefined
    }

    if (typeof toolCall.rawInvocation === "string" && toolCall.rawInvocation.trim().length > 0) {
        return toolCall.rawInvocation.trim()
    }

    const toolDefinition = resolveToolDefinitionForInvocation(toolDefinitions, toolCall.name)
    const parameters = toolCall.parameters ?? {}
    const passthroughParam = toolDefinition
        ? derivePassthroughParam(toolDefinition)
        : undefined

    if (passthroughParam) {
        const passthroughValue = readToolCallParameter(parameters, passthroughParam)
        if (typeof passthroughValue === "string" && passthroughValue.trim().length > 0) {
            return `${toolDefinition?.name ?? toolCall.name} ${passthroughValue.trim()}`
        }

        return toolDefinition?.name ?? toolCall.name
    }

    const parts = [toolDefinition?.name ?? toolCall.name]
    const positionalParams = toolDefinition
        ? derivePositionalParams(toolDefinition.parameters)
        : []
    const consumed = new Set<string>()

    for (const parameterName of positionalParams) {
        const value = readToolCallParameter(parameters, parameterName)
        consumed.add(parameterName)

        if (value === undefined || value === null) {
            continue
        }

        parts.push(formatCliArgument(value))
    }

    for (const [parameterName, value] of Object.entries(parameters)) {
        if (
            parameterName === "command" ||
            consumed.has(parameterName) ||
            value === undefined ||
            value === null
        ) {
            continue
        }

        const flagName = toFlagName(parameterName)
        if (typeof value === "boolean") {
            parts.push(value ? `--${flagName}` : `--no-${flagName}`)
            continue
        }

        parts.push(`--${flagName}`, formatCliArgument(value))
    }

    return parts.join(" ")
}

export function enrichToolResultWithUnixMetadata(
    toolCall: Pick<ToolCall, "name" | "kind" | "parameters" | "rawInvocation">,
    result: ToolResult,
    toolDefinitions: ToolDefinition[] = [],
): ToolResult {
    const command = formatToolCallAsUnixCommand(toolCall, toolDefinitions)
    if (!command) {
        return result
    }

    if (isBashToolCall(toolCall) || isBackgroundBashToolCall(toolCall)) {
        const primary = extractPrimaryCommandMetadata(command)
        return {
            ...result,
            command: typeof result.command === "string" && result.command.length > 0
                ? result.command
                : command,
            command_name:
                typeof result.command_name === "string" && result.command_name.length > 0
                    ? result.command_name
                    : primary.name ?? "bash",
            route:
                typeof result.route === "string" && result.route.length > 0
                    ? result.route
                    : toolCall.kind === "background_bash"
                        ? "background_bash"
                        : "bash",
        }
    }

    return {
        ...result,
        command: typeof result.command === "string" && result.command.length > 0
            ? result.command
            : command,
        command_name:
            typeof result.command_name === "string" && result.command_name.length > 0
                ? result.command_name
                : toolCall.name,
        route:
            typeof result.route === "string" && result.route.length > 0
                ? result.route
                : toolCall.name,
    }
}

export function extractCommandTargets(command: string): string[] {
    const chain = parseCommandChain(command)
    if (!chain) {
        return []
    }

    return chain.commands
        .map((tokens) => normalizeCommandName(tokens[0] ?? ""))
        .filter((value) => value.length > 0)
}

export function extractPrimaryCommandMetadata(
    command: string,
): { name?: string; path?: string; command?: string } {
    const chain = parseCommandChain(command)
    if (!chain || chain.commands.length === 0) {
        return { command }
    }

    const [firstCommand] = chain.commands
    const name = normalizeCommandName(firstCommand[0] ?? "")
    const args = firstCommand.slice(1)
    const firstPositional = args.find((token) => !token.startsWith("-"))

    return {
        name: name || undefined,
        path: firstPositional,
        command,
    }
}

export function tokenizeUnixCommand(
    command: string,
    options?: { allowTruncated?: boolean },
): string[] | null {
    return tokenizeCommand(command, options)
}

export function parseUnixToolCommand(
    tool: ToolDefinition,
    command: string,
    options?: {
        allowTruncated?: boolean
    },
): { ok: true; parameters: Record<string, unknown> } | { ok: false; error: string } {
    const descriptor = buildToolDescriptor(tool)

    if (descriptor.passthroughParam) {
        return parsePassthroughToolCommand(descriptor, command)
    }

    const tokens = tokenizeCommand(command, options)

    if (!tokens || tokens.length === 0) {
        return {
            ok: false,
            error: `${descriptor.command}: empty invocation. Usage: ${descriptor.usage}`,
        }
    }

    if (tokens[0] !== descriptor.command) {
        return {
            ok: false,
            error: `${descriptor.command}: invocation must start with ${descriptor.command}. Usage: ${descriptor.usage}`,
        }
    }

    const result = parseToolCommandArguments(descriptor, tokens.slice(1), options)

    // Fallback: when standard parsing fails, try extracting bracket-
    // balanced JSON values directly from the raw command string.  This
    // recovers from edge-case tokenisation failures (e.g. unquoted JSON
    // with special characters the tokenizer may split incorrectly).
    if (!result.ok) {
        const recovered = tryRecoverFlagParamsFromRawCommand(descriptor, command)
        if (recovered) {
            return recovered
        }
    }

    return result
}

export async function executeToolCallWithUnixSupport(
    runtime: CommandRuntime,
    toolCall: ToolCall,
): Promise<ToolResult> {
    const toolDefinitions = runtime.getToolDefinitions()

    if (isBashToolCall(toolCall)) {
        return executeUnixCommandString(
            typeof toolCall.parameters?.command === "string" ? toolCall.parameters.command : "",
            runtime,
        )
    }

    if (isBackgroundBashToolCall(toolCall)) {
        return executeBackgroundUnixCommandString(
            typeof toolCall.parameters?.command === "string" ? toolCall.parameters.command : "",
            runtime,
        )
    }

    let effectiveParameters = toolCall.parameters
    const definition = resolveToolDefinitionForInvocation(toolDefinitions, toolCall.name)

    if (
        definition &&
        typeof toolCall.rawInvocation === "string" &&
        toolCall.rawInvocation.trim().length > 0
    ) {
        const parsed = parseUnixToolCommand(definition, toolCall.rawInvocation, {
            allowTruncated: toolCall.truncated === true,
        })

        if (parsed.ok) {
            effectiveParameters = parsed.parameters as Record<string, any>
        } else {
            return enrichToolResultWithUnixMetadata(
                toolCall,
                {
                    success: false,
                    error: parsed.error,
                    stderr: parsed.error,
                    exit_code: 1,
                },
                toolDefinitions,
            )
        }
    }

    const result = await runtime.executeTool(toolCall.name, effectiveParameters)
    return enrichToolResultWithUnixMetadata(toolCall, result, toolDefinitions)
}

export async function executeUnixCommandString(
    command: string,
    runtime: CommandRuntime,
): Promise<ToolResult> {
    const trimmedCommand = command.trim()
    if (!trimmedCommand) {
        return {
            success: false,
            error: "Bash command is required",
            stderr: "Bash command is required",
            exit_code: 1,
            command: trimmedCommand,
            command_name: "bash",
            route: "bash",
        }
    }

    if (!runtime.executeBash) {
        return {
            success: false,
            error: "Bash execution is not available in this environment",
            stderr: "Bash execution is not available in this environment",
            exit_code: 1,
            command: trimmedCommand,
            command_name: "bash",
            route: "bash",
        }
    }

    const result = await runtime.executeBash(trimmedCommand)
    return enrichToolResultWithUnixMetadata(
        {
            name: "Bash",
            kind: "bash",
            parameters: { command: trimmedCommand },
        },
        result,
    )
}

export async function executeBackgroundUnixCommandString(
    command: string,
    runtime: CommandRuntime,
): Promise<ToolResult> {
    const trimmedCommand = command.trim()
    if (!trimmedCommand) {
        return {
            success: false,
            error: "Background bash command is required",
            stderr: "Background bash command is required",
            exit_code: 1,
            command: trimmedCommand,
            command_name: "background_bash",
            route: "background_bash",
        }
    }

    if (!runtime.executeBackgroundBash) {
        return {
            success: false,
            error: "Background bash execution is not available in this environment",
            stderr: "Background bash execution is not available in this environment",
            exit_code: 1,
            command: trimmedCommand,
            command_name: "background_bash",
            route: "background_bash",
        }
    }

    const result = await runtime.executeBackgroundBash(trimmedCommand)
    return enrichToolResultWithUnixMetadata(
        {
            name: "BackgroundBash",
            kind: "background_bash",
            parameters: { command: trimmedCommand },
        },
        result,
    )
}

export function deriveCommandName(name: string): string {
    if (name.includes("_")) {
        return name.toLowerCase()
    }

    return name
        .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
        .replace(/\s+/g, "-")
        .toLowerCase()
}

function buildToolDescriptor(tool: ToolDefinition): CommandDescriptor {
    const passthroughParam = derivePassthroughParam(tool)
    const { summary, details } = splitDescription(tool.description)

    return {
        command: tool.name,
        summary,
        details,
        usage: buildUsage(tool.name, tool.parameters, passthroughParam),
        example: tool.usageExample ?? buildToolUsageExample(tool),
        toolName: tool.name,
        parameters: tool.parameters,
        positionalParams: derivePositionalParams(tool.parameters),
        passthroughParam,
    }
}

function splitDescription(description: string): { summary: string; details?: string } {
    const match = description.match(/^(.+?[.!?])\s+([\s\S]+)$/)
    if (!match) {
        return { summary: description.trim() }
    }

    const summary = match[1].trim()
    const rest = match[2].trim()
    return rest.length > 0 ? { summary, details: rest } : { summary }
}

function parseToolCommandArguments(
    descriptor: CommandDescriptor,
    args: string[],
    _options?: {
        allowTruncated?: boolean
    },
): { ok: true; parameters: Record<string, unknown> } | { ok: false; error: string } {
    if (descriptor.passthroughParam) {
        const passthroughArgs = args.join(" ").trim()
        if (!passthroughArgs) {
            return {
                ok: false,
                error: `${descriptor.command}: missing required ${descriptor.passthroughParam}. Usage: ${descriptor.usage}`,
            }
        }

        return {
            ok: true,
            parameters: {
                [descriptor.passthroughParam]: passthroughArgs,
            },
        }
    }

    const parameters = descriptor.parameters
    const paramEntries = Object.entries(parameters)
    const parsed = parseCommandArgs(descriptor, args)

    if (parsed.error) {
        return {
            ok: false,
            error: parsed.error,
        }
    }

    if (parsed.unknownFlags.length > 0) {
        const firstUnknownFlag = parsed.unknownFlags[0]
        const positionalFlagHint = buildPositionalFlagError(descriptor, firstUnknownFlag)
        return {
            ok: false,
            error: positionalFlagHint
                ?? `${descriptor.command}: unknown flag ${firstUnknownFlag}. Usage: ${descriptor.usage}`,
        }
    }

    const params: Record<string, unknown> = { ...parsed.flags }
    const positional = parsed.positional
    const positionalParams = descriptor.positionalParams

    const assignmentStyleParameterError = positional
        .map((token) => buildAssignmentStyleParameterError(descriptor, token))
        .find((error): error is string => typeof error === "string")

    if (assignmentStyleParameterError) {
        return {
            ok: false,
            error: assignmentStyleParameterError,
        }
    }

    if (positional.length > 0) {
        if (positionalParams.length === 0) {
            return {
                ok: false,
                error: `${descriptor.command}: unexpected positional arguments. Usage: ${descriptor.usage}`,
            }
        }

        const lastPositionalName = positionalParams[positionalParams.length - 1]
        const lastPositionalDefinition = parameters[lastPositionalName]
        const joinRemainder =
            positional.length > positionalParams.length &&
            lastPositionalDefinition &&
            isStringType(lastPositionalDefinition.type)

        for (let index = 0; index < positionalParams.length; index++) {
            const paramName = positionalParams[index]
            const definition = parameters[paramName]
            if (!definition || params[paramName] !== undefined) {
                continue
            }

            if (joinRemainder && index === positionalParams.length - 1) {
                params[paramName] = coerceCliValue(positional.slice(index).join(" "), definition)
                break
            }

            const value = positional[index]
            if (value !== undefined) {
                params[paramName] = coerceCliValue(value, definition)
            }
        }

        if (!joinRemainder && positional.length > positionalParams.length) {
            return {
                ok: false,
                error: `${descriptor.command}: too many positional arguments. Usage: ${descriptor.usage}`,
            }
        }
    }

    const missingRequired = paramEntries.filter(
        ([name, definition]) => definition.required && params[name] === undefined,
    )

    if (missingRequired.length > 0) {
        return {
            ok: false,
            error: `${descriptor.command}: missing required ${missingRequired.map(([name]) => name).join(", ")}. Usage: ${descriptor.usage}`,
        }
    }

    return { ok: true, parameters: params }
}

function parseCommandArgs(
    descriptor: CommandDescriptor,
    args: string[],
): ParsedCommandArgs {
    const parameterDefinitions = descriptor.parameters
    const flags: Record<string, unknown> = {}
    const positional: string[] = []
    const unknownFlags: string[] = []
    const flagParameters = new Map<string, [string, ToolParameterDefinition]>()

    for (const [parameterName, definition] of Object.entries(parameterDefinitions)) {
        flagParameters.set(toFlagName(parameterName), [parameterName, definition])
    }

    for (let index = 0; index < args.length; index++) {
        const token = args[index]

        if (token === "--") {
            positional.push(...args.slice(index + 1))
            break
        }

        if (token === "-" || !token.startsWith("-")) {
            positional.push(token)
            continue
        }

        if (!token.startsWith("--")) {
            unknownFlags.push(token)
            continue
        }

        if (token.startsWith("--no-")) {
            const flagName = token.slice(5)
            const parameterEntry = flagParameters.get(flagName)
            if (!parameterEntry || !isBooleanType(parameterEntry[1].type)) {
                unknownFlags.push(token)
                continue
            }

            flags[parameterEntry[0]] = false
            continue
        }

        const equalsIndex = token.indexOf("=")
        const flagToken = equalsIndex >= 0 ? token.slice(0, equalsIndex) : token
        const flagName = flagToken.slice(2)
        const parameterEntry = flagParameters.get(flagName)

        if (!parameterEntry) {
            unknownFlags.push(flagToken)
            continue
        }

        if (equalsIndex >= 0) {
            return {
                positional,
                flags,
                unknownFlags,
                error: `${descriptor.command}: ${token} is not valid. Use ${flagToken} <value> with a space-separated value. Usage: ${descriptor.usage}`,
            }
        }

        const [parameterName, definition] = parameterEntry
        if (descriptor.positionalParams.includes(parameterName)) {
            return {
                positional,
                flags,
                unknownFlags,
                error: buildRequiredPositionalParameterFlagError(
                    descriptor,
                    flagToken,
                    parameterName,
                ),
            }
        }

        if (isBooleanType(definition.type)) {
            flags[parameterName] = true
            continue
        }

        const rawValue = args[index + 1]

        if (rawValue === undefined || rawValue.length === 0) {
            return {
                positional,
                flags,
                unknownFlags,
                error: `${descriptor.command}: ${flagToken} requires a value. Usage: ${descriptor.usage}`,
            }
        }

        // For array/object types, if the value starts with [ or { but
        // isn't bracket-balanced (tokenizer split the JSON), greedily
        // consume subsequent tokens until brackets close.
        const expectedTypes = Array.isArray(definition.type)
            ? definition.type
            : [definition.type]
        if (
            (expectedTypes.includes("array") || expectedTypes.includes("object"))
            && /^[\[{]/.test(rawValue)
            && !isBracketBalanced(rawValue)
        ) {
            let aggregated = rawValue
            let endIdx = index + 1
            while (!isBracketBalanced(aggregated) && endIdx + 1 < args.length) {
                endIdx++
                aggregated += " " + args[endIdx]
            }
            flags[parameterName] = coerceCliValue(aggregated, definition)
            index = endIdx
            continue
        }

        flags[parameterName] = coerceCliValue(rawValue, definition)
        index += 1
    }

    return {
        positional,
        flags,
        unknownFlags,
    }
}

function buildUsage(
    command: string,
    parameters: Record<string, ToolParameterDefinition>,
    passthroughParam?: string,
): string {
    if (passthroughParam) {
        return `${command} <${passthroughParam}...>`
    }

    const parts = [command]
    const positional = derivePositionalParams(parameters)

    for (const name of positional) {
        const definition = parameters[name]
        if (definition?.required) {
            parts.push(`<${name}>`)
        } else {
            parts.push(`[${name}]`)
        }
    }

    for (const [name, definition] of Object.entries(parameters)) {
        if (positional.includes(name)) {
            continue
        }

        const flag = `--${toFlagName(name)}`
        if (definition.required) {
            parts.push(isBooleanType(definition.type) ? flag : `${flag} <${formatType(definition.type)}>`)
        } else {
            parts.push(isBooleanType(definition.type) ? `[${flag}]` : `[${flag} <${formatType(definition.type)}>]`)
        }
    }

    return parts.join(" ")
}

function derivePositionalParams(
    parameters: Record<string, ToolParameterDefinition>,
): string[] {
    return Object.entries(parameters)
        .filter(([, definition]) => isPositionalParameterDefinition(definition))
        .map(([name]) => name)
}

function isPositionalParameterDefinition(
    definition: ToolParameterDefinition,
): boolean {
    if (!definition.required || isBooleanType(definition.type)) {
        return false
    }

    return definition.positional !== false
}

function derivePassthroughParam(tool: ToolDefinition): string | undefined {
    if (!tool.metadata?.capabilities?.includes("raw_command_args")) {
        return undefined
    }

    const parameterEntries = Object.entries(tool.parameters)
    if (parameterEntries.length !== 1) {
        return undefined
    }

    const [parameterName, definition] = parameterEntries[0]
    if (!definition.required || !isStringType(definition.type)) {
        return undefined
    }

    return parameterName
}

function parsePassthroughToolCommand(
    descriptor: CommandDescriptor,
    command: string,
): { ok: true; parameters: Record<string, unknown> } | { ok: false; error: string } {
    const passthroughParam = descriptor.passthroughParam
    if (!passthroughParam) {
        return {
            ok: false,
            error: `${descriptor.command}: internal passthrough parsing error. Usage: ${descriptor.usage}`,
        }
    }

    const trimmed = command.trim()
    if (!trimmed.startsWith(descriptor.command)) {
        return {
            ok: false,
            error: `${descriptor.command}: invocation must start with ${descriptor.command}. Usage: ${descriptor.usage}`,
        }
    }

    const nextCharacter = trimmed[descriptor.command.length]
    if (nextCharacter && !/\s/.test(nextCharacter)) {
        return {
            ok: false,
            error: `${descriptor.command}: invocation must start with ${descriptor.command}. Usage: ${descriptor.usage}`,
        }
    }

    const rawArguments = trimmed.slice(descriptor.command.length).trim()
    if (!rawArguments) {
        return {
            ok: false,
            error: `${descriptor.command}: missing required ${passthroughParam}. Usage: ${descriptor.usage}`,
        }
    }

    const passthroughWrapperMatch = rawArguments.match(/^--([a-zA-Z][\w-]*)(?:\s|=)/)
    if (passthroughWrapperMatch) {
        const wrapperFlag = `--${passthroughWrapperMatch[1]}`
        const acceptedWrapperFlags = new Set([`--${toFlagName(passthroughParam)}`])

        if (acceptedWrapperFlags.has(wrapperFlag)) {
            const afterFlag = rawArguments.slice(passthroughWrapperMatch[0].length).trim()
            if (afterFlag) {
                return { ok: true, parameters: { [passthroughParam]: afterFlag } }
            }
            return {
                ok: false,
                error: `${descriptor.command}: missing required ${passthroughParam} after ${wrapperFlag}. Usage: ${descriptor.usage}`,
            }
        }
        if (wrapperFlag === "--options") {
            return {
                ok: false,
                error: `${descriptor.command}: ${wrapperFlag} is not valid for this passthrough tool. Provide the raw argument string positionally after the tool name. Usage: ${descriptor.usage}`,
            }
        }
    }

    return {
        ok: true,
        parameters: {
            [passthroughParam]: rawArguments,
        },
    }
}

function buildPositionalFlagError(
    descriptor: CommandDescriptor,
    flagToken: string,
): string | undefined {
    if (!flagToken.startsWith("--")) {
        return undefined
    }

    const normalizedFlagName = flagToken.startsWith("--no-")
        ? flagToken.slice(5)
        : flagToken.slice(2)

    for (const positionalName of descriptor.positionalParams) {
        if (toFlagName(positionalName) === normalizedFlagName) {
            return buildRequiredPositionalParameterFlagError(
                descriptor,
                flagToken,
                positionalName,
            )
        }
    }

    return undefined
}

function buildRequiredPositionalParameterFlagError(
    descriptor: CommandDescriptor,
    flagToken: string,
    parameterName: string,
): string {
    return `${descriptor.command}: ${flagToken} is not valid for required positional parameter ${parameterName}. Provide that value positionally. Usage: ${descriptor.usage}`
}

function buildAssignmentStyleParameterError(
    descriptor: CommandDescriptor,
    token: string,
): string | undefined {
    const parameterMatch = matchAssignmentStyleParameter(descriptor, token)
    if (!parameterMatch) {
        return undefined
    }

    const { parameterName, definition, rawName } = parameterMatch
    if (descriptor.positionalParams.includes(parameterName)) {
        return `${descriptor.command}: ${token} is not valid for required positional parameter ${parameterName}. Provide that value positionally without ${rawName}=. Usage: ${descriptor.usage}`
    }

    const flagName = `--${toFlagName(parameterName)}`
    if (isBooleanType(definition.type)) {
        return `${descriptor.command}: ${token} is not valid for parameter ${parameterName}. Use ${flagName} or --no-${toFlagName(parameterName)}. Usage: ${descriptor.usage}`
    }

    return `${descriptor.command}: ${token} is not valid for parameter ${parameterName}. Use ${flagName} <${formatType(definition.type)}>. Usage: ${descriptor.usage}`
}

function matchAssignmentStyleParameter(
    descriptor: CommandDescriptor,
    token: string,
): {
    parameterName: string
    definition: ToolParameterDefinition
    rawName: string
    rawValue: string
} | null {
    const match = token.match(/^([A-Za-z_][\w-]*)=(.+)$/)
    if (!match) {
        return null
    }

    const [, rawName, rawValue] = match
    const parameterEntry = Object.entries(descriptor.parameters).find(([name]) => (
        name === rawName || toFlagName(name) === rawName
    ))

    if (!parameterEntry) {
        return null
    }

    const [parameterName, definition] = parameterEntry

    return {
        parameterName,
        definition,
        rawName,
        rawValue,
    }
}

function parseCommandChain(
    command: string,
): { commands: string[][]; operators: ChainOperator[] } | null {
    const tokens = tokenizeCommand(command)
    if (!tokens) {
        return null
    }
    if (tokens.length === 0) {
        return { commands: [], operators: [] }
    }

    const commands: string[][] = []
    const operators: ChainOperator[] = []
    let current: string[] = []

    for (const token of tokens) {
        if (isOperatorToken(token)) {
            if (current.length === 0) {
                return null
            }
            commands.push(current)
            operators.push(token)
            current = []
            continue
        }
        current.push(token)
    }

    if (current.length === 0) {
        return null
    }

    commands.push(current)
    return { commands, operators }
}

function tokenizeCommand(
    command: string,
    options?: { allowTruncated?: boolean },
): string[] | null {
    const tokens: string[] = []
    let current = ""
    let quote: '"' | "'" | null = null
    let escaping = false
    let bracketDepth = 0
    const allowTruncated = options?.allowTruncated === true

    const flush = () => {
        if (current.length > 0) {
            tokens.push(current)
            current = ""
        }
    }

    for (let index = 0; index < command.length; index++) {
        const char = command[index]
        const next = command[index + 1]

        if (escaping) {
            current += shouldConsumeEscape(char, quote)
                ? char
                : `\\${char}`
            escaping = false
            continue
        }

        if (char === "\\") {
            escaping = true
            continue
        }

        if (quote) {
            if (char === quote) {
                quote = null
                // Inside a bracket literal, preserve the quotes so the
                // resulting token is valid JSON.
                if (bracketDepth > 0) {
                    current += char
                }
            } else {
                current += char
            }
            continue
        }

        if (char === '"' || char === "'") {
            // Inside a bracket literal, preserve opening quotes.
            if (bracketDepth > 0) {
                current += char
            }
            quote = char
            continue
        }

        // Bracket-balanced JSON literal: [ ... ] or { ... }
        if (bracketDepth > 0) {
            if (char === "[" || char === "{") {
                bracketDepth++
            } else if (char === "]" || char === "}") {
                bracketDepth--
            }
            current += char
            continue
        }

        if (char === "[" || char === "{") {
            bracketDepth = 1
            current += char
            continue
        }

        if (char === "&" && next === "&") {
            flush()
            tokens.push("&&")
            index++
            continue
        }

        if (char === "|" && next === "|") {
            flush()
            tokens.push("||")
            index++
            continue
        }

        if (char === "|" || char === ";") {
            flush()
            tokens.push(char)
            continue
        }

        if (/\s/.test(char)) {
            flush()
            continue
        }

        current += char
    }

    if (escaping || quote || (bracketDepth > 0 && !allowTruncated)) {
        if (!allowTruncated) {
            return null
        }

        if (escaping) {
            current += "\\"
        }

        flush()
        return tokens
    }

    flush()
    return tokens
}

function shouldConsumeEscape(
    char: string,
    quote: '"' | "'" | null,
): boolean {
    if (quote === "'") {
        return char === "'" || char === "\\"
    }

    if (quote === '"') {
        return char === '"' || char === "\\"
    }

    return /\s/.test(char) || char === '"' || char === "'" || char === "\\" || char === "|" || char === "&" || char === ";"
}

/**
 * Returns true when every `[` / `{` in the string has a matching `]` / `}`.
 * Respects double-quoted JSON strings (escaped quotes are handled).
 */
function isBracketBalanced(value: string): boolean {
    let depth = 0
    let inString = false
    let escaping = false
    for (const char of value) {
        if (escaping) {
            escaping = false
            continue
        }
        if (char === "\\") {
            escaping = true
            continue
        }
        if (char === '"') {
            inString = !inString
            continue
        }
        if (inString) continue
        if (char === "[" || char === "{") depth++
        else if (char === "]" || char === "}") depth--
    }
    return depth === 0
}

/**
 * Last-resort recovery: scan the raw command string for `--flag`
 * followed by a bracket-balanced JSON value.  This bypasses the
 * tokenizer entirely and handles cases where the tokenizer splits
 * or corrupts inline JSON.
 */
function tryRecoverFlagParamsFromRawCommand(
    descriptor: CommandDescriptor,
    command: string,
): { ok: true; parameters: Record<string, unknown> } | null {
    const params: Record<string, unknown> = {}
    let recovered = false

    for (const [paramName, definition] of Object.entries(descriptor.parameters)) {
        const expectedTypes = Array.isArray(definition.type) ? definition.type : [definition.type]
        if (!expectedTypes.includes("array") && !expectedTypes.includes("object")) {
            continue
        }

        const flagName = `--${toFlagName(paramName)}`
        const flagIdx = command.indexOf(flagName)
        if (flagIdx < 0) continue

        // Find the start of the JSON value after the flag
        let start = flagIdx + flagName.length
        // Skip whitespace between flag and value
        while (start < command.length && /\s/.test(command[start])) start++
        if (start >= command.length) continue

        const opener = command[start]
        if (opener !== "[" && opener !== "{") continue

        // Extract bracket-balanced substring
        const jsonStr = extractBracketBalanced(command, start)
        if (!jsonStr) continue

        try {
            params[paramName] = JSON.parse(jsonStr)
            recovered = true
        } catch {
            // Not valid JSON, skip
        }
    }

    if (!recovered) return null

    // Check required params that weren't recovered
    for (const [name, definition] of Object.entries(descriptor.parameters)) {
        if (definition.required && params[name] === undefined) {
            return null
        }
    }

    return { ok: true, parameters: params }
}

/**
 * Extract a bracket-balanced substring starting at `start`.
 * Respects double-quoted strings inside the JSON.
 */
function extractBracketBalanced(str: string, start: number): string | null {
    let depth = 0
    let inString = false
    let escaping = false

    for (let i = start; i < str.length; i++) {
        const char = str[i]
        if (escaping) {
            escaping = false
            continue
        }
        if (char === "\\") {
            escaping = true
            continue
        }
        if (char === '"') {
            inString = !inString
            continue
        }
        if (inString) continue
        if (char === "[" || char === "{") depth++
        else if (char === "]" || char === "}") {
            depth--
            if (depth === 0) {
                return str.slice(start, i + 1)
            }
        }
    }
    return null
}

function normalizeCommandName(name: string): string {
    return name.trim().toLowerCase()
}

function isOperatorToken(token: string): token is ChainOperator {
    return token === "|" || token === "&&" || token === "||" || token === ";"
}

function readToolCallParameter(
    parameters: Record<string, any>,
    parameterName: string,
): unknown {
    if (parameters[parameterName] !== undefined) {
        return parameters[parameterName]
    }

    if (parameterName === "path" && parameters.file_path !== undefined) {
        return parameters.file_path
    }

    if (parameterName === "taskId" && parameters.task_id !== undefined) {
        return parameters.task_id
    }

    return undefined
}

function formatCliArgument(value: unknown): string {
    if (typeof value === "string") {
        return quoteCliArgument(value)
    }

    if (typeof value === "number" || typeof value === "bigint") {
        return String(value)
    }

    if (typeof value === "boolean") {
        return value ? "true" : "false"
    }

    return quoteCliArgument(JSON.stringify(value))
}

function quoteCliArgument(value: string): string {
    if (value.length === 0) {
        return '""'
    }

    if (/^[^\s"'|&;]+$/.test(value)) {
        return value
    }

    return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`
}

function toFlagName(name: string): string {
    return name.replace(/_/g, "-").toLowerCase()
}

function formatType(type: string | string[]): string {
    return Array.isArray(type) ? type.join("|") : type
}

function isBooleanType(type: string | string[]): boolean {
    return (Array.isArray(type) ? type : [type]).includes("boolean")
}

function isStringType(type: string | string[]): boolean {
    return (Array.isArray(type) ? type : [type]).includes("string")
}

function coerceCliValue(
    value: string,
    definition: ToolParameterDefinition,
): string | number | boolean | Record<string, unknown> | unknown[] {
    const expectedTypes = Array.isArray(definition.type)
        ? definition.type
        : [definition.type]

    if (expectedTypes.includes("boolean")) {
        if (value === "true") return true
        if (value === "false") return false
    }

    if (expectedTypes.includes("integer")) {
        const parsed = Number(value)
        if (Number.isInteger(parsed)) return parsed
    }

    if (expectedTypes.includes("number")) {
        const parsed = Number(value)
        if (!Number.isNaN(parsed)) return parsed
    }

    if (expectedTypes.includes("array") || expectedTypes.includes("object")) {
        try {
            return JSON.parse(value) as Record<string, unknown> | unknown[]
        } catch {
            if (expectedTypes.includes("array")) {
                const bracketArray = parseBracketArrayLiteral(value, definition)
                if (bracketArray) {
                    return bracketArray
                }

                return value.split(",").map((entry) => entry.trim()).filter(Boolean)
            }
        }
    }

    return value
}

function parseBracketArrayLiteral(
    value: string,
    definition: ToolParameterDefinition,
): unknown[] | null {
    if (!value.startsWith("[") || !value.endsWith("]")) {
        return null
    }

    const inner = value.slice(1, -1).trim()
    if (inner.length === 0) {
        return []
    }

    return inner
        .split(",")
        .map((entry) => entry.trim())
        .filter(Boolean)
        .map((entry) => coerceArrayItemValue(stripWrappingQuotes(entry), definition))
}

function coerceArrayItemValue(
    value: string,
    definition: ToolParameterDefinition,
): unknown {
    const itemType = definition.items?.type
    if (!itemType) {
        return value
    }

    const itemDefinition: ToolParameterDefinition = {
        type: itemType,
        description: definition.items?.description ?? definition.description,
        items: definition.items?.items,
        properties: definition.items?.properties,
        additionalProperties: definition.items?.additionalProperties,
        default: definition.items?.default,
    }

    return coerceCliValue(value, itemDefinition)
}

function stripWrappingQuotes(value: string): string {
    if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
    ) {
        return value.slice(1, -1)
    }

    return value
}
