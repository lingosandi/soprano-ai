import type {
    LoopOutcome,
    ToolCall,
    ToolDefinition,
    ToolResult,
} from "./types"
import {
    enrichToolResultWithUnixMetadata,
    executeBackgroundUnixCommandString,
    executeUnixCommandString,
    parseUnixToolCommand,
    tokenizeUnixCommand,
    type CommandRuntime,
} from "./unix-tooling"
import { decodeEscapedShellText } from "./shell-escape-normalization"

export type RunInvocationTarget = "tool" | "bash" | "background_bash"

export interface ExtractedRunInvocationLike {
    target: RunInvocationTarget
    command: string
    truncated: boolean
}

export interface RunCommandParseResult {
    outcome?: LoopOutcome
    selectedToolCall?: ToolCall
    selectedToolCallCount?: number
}

export interface RunCommandPromptAvailability {
    bashAvailable: boolean
    backgroundBashAvailable: boolean
}

const RUN_TOOL_ALIAS_VALIDATION_ERROR_PREFIX = "VALIDATION_ERROR:"

function normalizeRunToolName(name: string): string {
    return name.replace(/[_\-\s]+/g, "").toLowerCase()
}

function buildMisroutedRunToolValidationError(toolName: string): Error | null {
    const normalizedToolName = normalizeRunToolName(toolName)

    if (normalizedToolName === "bash") {
        return new Error(
            `${RUN_TOOL_ALIAS_VALIDATION_ERROR_PREFIX} Bash is not a registered tool name. You wrote a tool payload with a bash command. Use the ---bash--- header instead.`,
        )
    }

    if (normalizedToolName === "backgroundbash") {
        return new Error(
            `${RUN_TOOL_ALIAS_VALIDATION_ERROR_PREFIX} BackgroundBash is not a registered tool name. Use the ---background_bash--- header instead of a tool payload.`,
        )
    }

    return null
}

export type BackgroundBashAction = "start" | "status" | "logs" | "stop"

export interface ParsedBackgroundBashCommand {
    action: BackgroundBashAction
    name: string
    rawCommand: string
    startCommand?: string
    port?: number
    tailBytes?: number
}

export const DEFAULT_ALLOWED_RUN_TARGETS: readonly RunInvocationTarget[] = [
    "tool",
    "bash",
]

export const SUPPORTED_RUN_TARGETS: readonly RunInvocationTarget[] = [
    "tool",
    "bash",
    "background_bash",
]

function normalizeShellPromptCommand(command: string): string {
    const trimmedCommand = decodeEscapedMultilineCommand(command.trim())
    if (!/^\$\s+/.test(trimmedCommand)) {
        return stripTranscriptArtifactsAfterHeredoc(trimmedCommand)
    }

    const lines = trimmedCommand.split(/\r?\n/)
    const normalizedLines = [...lines]
    normalizedLines[0] = normalizedLines[0]!.replace(/^\$\s+/, "")

    const continuationLines = normalizedLines.slice(1)
    if (
        continuationLines.length > 0
        && continuationLines.every((line) => line.trim().length === 0 || /^>\s?/.test(line))
    ) {
        for (let index = 1; index < normalizedLines.length; index += 1) {
            normalizedLines[index] = normalizedLines[index]!.replace(/^>\s?/, "")
        }
    }

    return stripTranscriptArtifactsAfterHeredoc(normalizedLines.join("\n").trim())
}

function decodeEscapedMultilineCommand(command: string): string {
    if (command.includes("\n") || command.includes("\r")) {
        return command
    }

    return decodeEscapedShellText(command)
}

function stripTranscriptArtifactsAfterHeredoc(command: string): string {
    const lines = command.split(/\r?\n/)
    const heredocInfo = findFirstHeredoc(lines)
    if (!heredocInfo) {
        return command
    }

    const { terminator, startLineIndex, terminatorLineIndex } = heredocInfo

    if (terminatorLineIndex === -1) {
        const firstTranscriptArtifactLineIndex = lines.findIndex(
            (line, index) =>
                index > startLineIndex
                && isTranscriptArtifactLine(line?.trim() ?? ""),
        )

        if (firstTranscriptArtifactLineIndex !== -1) {
            return [
                ...lines.slice(0, firstTranscriptArtifactLineIndex),
                terminator,
            ].join("\n").trim()
        }

        return [...lines, terminator].join("\n").trim()
    }

    const normalizedLines = collapseDuplicateHeredocTerminators(
        lines,
        terminator,
        terminatorLineIndex,
    )
    const normalizedTerminatorLineIndex = findTerminatorLineIndex(
        normalizedLines,
        terminator,
        startLineIndex,
    )

    const firstTranscriptArtifactLineIndex = normalizedLines.findIndex(
        (line, index) =>
            index > normalizedTerminatorLineIndex
            && isTranscriptArtifactLine(line?.trim() ?? ""),
    )

    if (firstTranscriptArtifactLineIndex !== -1) {
        return normalizedLines.slice(0, normalizedTerminatorLineIndex + 1).join("\n").trim()
    }

    return normalizedLines.join("\n").trim()
}

function findFirstHeredoc(lines: string[]): {
    terminator: string
    startLineIndex: number
    terminatorLineIndex: number
} | null {
    for (let index = 0; index < lines.length; index += 1) {
        const match = lines[index]?.match(/<<-?\s*(['"]?)([A-Za-z_][A-Za-z0-9_]*)\1/)
        if (!match?.[2]) {
            continue
        }

        const terminator = match[2]
        let terminatorLineIndex = -1

        for (let lineIndex = index + 1; lineIndex < lines.length; lineIndex += 1) {
            if (lines[lineIndex]?.trim() === terminator) {
                terminatorLineIndex = lineIndex
                break
            }
        }

        return {
            terminator,
            startLineIndex: index,
            terminatorLineIndex,
        }
    }

    return null
}

function findTerminatorLineIndex(
    lines: string[],
    terminator: string,
    startLineIndex: number,
): number {
    for (let lineIndex = startLineIndex + 1; lineIndex < lines.length; lineIndex += 1) {
        if (lines[lineIndex]?.trim() === terminator) {
            return lineIndex
        }
    }

    return -1
}

function collapseDuplicateHeredocTerminators(
    lines: string[],
    terminator: string,
    terminatorLineIndex: number,
): string[] {
    const normalizedLines = [...lines]
    let duplicateLineIndex = terminatorLineIndex + 1

    while (normalizedLines[duplicateLineIndex]?.trim() === terminator) {
        normalizedLines.splice(duplicateLineIndex, 1)
    }

    return normalizedLines
}

function isTranscriptArtifactLine(line: string): boolean {
    return /^\[(?:stderr|meta|exit)\]$/i.test(line)
        || /^\[[A-Z_]+(?::[^\]]+)?\]$/.test(line)
        || /^\$(?:\s|$)/.test(line)
        || /^[^\s@]+@.+[$#>]$/.test(line)
}

export function createBackgroundBashDefinition({
    description,
    portDescription,
}: {
    description: string
    portDescription: string
}): ToolDefinition {
    return {
        name: "BackgroundBash",
        description,
        usageExample: 'BackgroundBash start hello --command "bun server.js" --port 3000',
        parameters: {
            action: {
                type: "string",
                description: "Background process action: start, status, stop, or logs.",
                required: true,
                positional: true,
                enum: ["start", "status", "stop", "logs"] as const,
            },
            name: {
                type: "string",
                description: "Stable process name, such as hello or dev-server.",
                required: true,
                positional: true,
            },
            command: {
                type: "string",
                description: "Shell command to start the process. Required only for start.",
            },
            port: {
                type: "number",
                description: portDescription,
            },
            tail_bytes: {
                type: "number",
                description: "Optional number of trailing log bytes to return for status or logs.",
            },
        },
    }
}

export abstract class RunCommand {
    readonly target: RunInvocationTarget

    protected constructor(target: RunInvocationTarget) {
        this.target = target
    }

    abstract isAvailable(runtime: CommandRuntime): boolean

    abstract parseInvocation(
        invocation: ExtractedRunInvocationLike,
        options?: { allowTruncated?: boolean },
    ): RunCommandParseResult | null

    abstract execute(
        toolCall: ToolCall,
        runtime: CommandRuntime,
    ): Promise<ToolResult>

    matchesToolCall(toolCall: ToolCall): boolean {
        if (toolCall.kind) {
            return toolCall.kind === this.target
        }

        return this.target === "tool"
    }
}

export class ToolRunCommand extends RunCommand {
    constructor() {
        super("tool")
    }

    isAvailable(runtime: CommandRuntime): boolean {
        return runtime.getToolDefinitions().length > 0
    }

    parseInvocation(
        invocation: ExtractedRunInvocationLike,
        options?: { allowTruncated?: boolean },
    ): RunCommandParseResult | null {
        const tokens = tokenizeUnixCommand(invocation.command, {
            allowTruncated: options?.allowTruncated === true && invocation.truncated,
        })
        const toolName = tokens?.[0]?.trim()
        if (!toolName) {
            return null
        }

        const aliasError = buildMisroutedRunToolValidationError(toolName)
        if (aliasError) {
            throw aliasError
        }

        return {
            outcome: "continue",
            selectedToolCall: {
                kind: "tool",
                name: toolName,
                parameters: {},
                rawInvocation: invocation.command,
                ...(invocation.truncated ? { truncated: true } : {}),
            },
            selectedToolCallCount: 1,
        }
    }

    async execute(
        toolCall: ToolCall,
        runtime: CommandRuntime,
    ): Promise<ToolResult> {
        const toolDefinitions = runtime.getToolDefinitions()
        const definition = toolDefinitions.find((tool) => tool.name === toolCall.name)
        let effectiveParameters = toolCall.parameters

        if (
            definition
            && typeof toolCall.rawInvocation === "string"
            && toolCall.rawInvocation.trim().length > 0
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
}

export class BashRunCommand extends RunCommand {
    constructor() {
        super("bash")
    }

    protected getCommand(toolCall: ToolCall): string {
        return typeof toolCall.parameters?.command === "string"
            ? toolCall.parameters.command
            : ""
    }

    isAvailable(runtime: CommandRuntime): boolean {
        return typeof runtime.executeBash === "function"
    }

    parseInvocation(invocation: ExtractedRunInvocationLike): RunCommandParseResult | null {
        const normalizedCommand = normalizeShellPromptCommand(invocation.command)
        const tokens = tokenizeUnixCommand(normalizedCommand, {
            allowTruncated: invocation.truncated,
        })
        if (!tokens || tokens.length === 0) {
            return null
        }

        if (tokens[0] === "exit") {
            if (tokens.length === 2 && tokens[1] === "0") {
                return { outcome: "success" }
            }

            if (tokens.length === 2 && tokens[1] === "1") {
                return { outcome: "failure" }
            }

            return null
        }

        return {
            outcome: "continue",
            selectedToolCall: {
                kind: "bash",
                name: "Bash",
                parameters: { command: normalizedCommand },
                ...(invocation.truncated ? { truncated: true } : {}),
            },
            selectedToolCallCount: 1,
        }
    }

    protected async executeCommand(
        command: string,
        runtime?: CommandRuntime,
    ): Promise<ToolResult> {
        if (!runtime) {
            throw new Error("Bash runtime is required")
        }

        return executeUnixCommandString(command, runtime)
    }

    executeRaw(
        command: string,
        runtime?: CommandRuntime,
    ): Promise<ToolResult> {
        return this.executeCommand(command, runtime)
    }

    async execute(
        toolCall: ToolCall,
        runtime: CommandRuntime,
    ): Promise<ToolResult> {
        return enrichToolResultWithUnixMetadata(
            toolCall,
            await this.executeRaw(this.getCommand(toolCall), runtime),
        )
    }
}

export class BackgroundBashRunCommand extends RunCommand {
    constructor() {
        super("background_bash")
    }

    protected getCommand(toolCall: ToolCall): string {
        return typeof toolCall.parameters?.command === "string"
            ? toolCall.parameters.command
            : ""
    }

    isAvailable(runtime: CommandRuntime): boolean {
        return typeof runtime.executeBackgroundBash === "function"
    }

    parseInvocation(invocation: ExtractedRunInvocationLike): RunCommandParseResult | null {
        const normalizedCommand = normalizeShellPromptCommand(invocation.command)

        if (!normalizedCommand) {
            return null
        }

        return {
            outcome: "continue",
            selectedToolCall: {
                kind: "background_bash",
                name: "BackgroundBash",
                parameters: { command: normalizedCommand },
                rawInvocation: normalizedCommand,
                ...(invocation.truncated ? { truncated: true } : {}),
            },
            selectedToolCallCount: 1,
        }
    }

    protected async executeCommand(
        command: string,
        runtime?: CommandRuntime,
    ): Promise<ToolResult> {
        if (!runtime) {
            throw new Error("Background bash runtime is required")
        }

        return executeBackgroundUnixCommandString(command, runtime)
    }

    executeRaw(
        command: string,
        runtime?: CommandRuntime,
    ): Promise<ToolResult> {
        return this.executeCommand(command, runtime)
    }

    protected buildBackgroundBashErrorResult(command: string, error: string): ToolResult {
        return {
            success: false,
            command,
            error,
            stderr: error,
            exit_code: 1,
            route: "background_bash",
        }
    }

    protected parseBackgroundBashCommand(
        definition: ToolDefinition,
        command: string,
    ):
        | { ok: true; value: ParsedBackgroundBashCommand }
        | { ok: false; result: ToolResult } {
        const parsed = parseUnixToolCommand(definition, `BackgroundBash ${command}`)
        if (!parsed.ok) {
            return {
                ok: false,
                result: this.buildBackgroundBashErrorResult(command, parsed.error),
            }
        }

        const parameters = parsed.parameters as Record<string, unknown>
        const action = String(parameters.action || "") as BackgroundBashAction
        const name = String(parameters.name || "").trim()
        const startCommand = typeof parameters.command === "string"
            ? parameters.command.trim()
            : ""
        const port = typeof parameters.port === "number" ? parameters.port : undefined
        const tailBytes = typeof parameters.tail_bytes === "number"
            ? parameters.tail_bytes
            : undefined

        if (!name) {
            return {
                ok: false,
                result: this.buildBackgroundBashErrorResult(
                    command,
                    "BackgroundBash: name is required",
                ),
            }
        }

        if (action === "start" && !startCommand) {
            return {
                ok: false,
                result: this.buildBackgroundBashErrorResult(
                    command,
                    "BackgroundBash: command is required for start. Usage: BackgroundBash start <name> --command <string> [--port <number>]",
                ),
            }
        }

        return {
            ok: true,
            value: {
                action,
                name,
                rawCommand: command,
                ...(startCommand ? { startCommand } : {}),
                ...(typeof port === "number" ? { port } : {}),
                ...(typeof tailBytes === "number" ? { tailBytes } : {}),
            },
        }
    }

    async execute(
        toolCall: ToolCall,
        runtime: CommandRuntime,
    ): Promise<ToolResult> {
        return enrichToolResultWithUnixMetadata(
            toolCall,
            await this.executeRaw(this.getCommand(toolCall), runtime),
        )
    }
}

export class RunCommandRegistry {
    private readonly commands: readonly RunCommand[]

    constructor(commands: readonly RunCommand[]) {
        this.commands = commands
    }

    getCommand(target: RunInvocationTarget): RunCommand | undefined {
        return this.commands.find((command) => command.target === target)
    }

    getSupportedTargets(): readonly RunInvocationTarget[] {
        return this.commands.map((command) => command.target)
    }

    getAllowedTargets(runtime: CommandRuntime): readonly RunInvocationTarget[] {
        return this.commands
            .filter((command) => command.isAvailable(runtime))
            .map((command) => command.target)
    }

    getPromptAvailability(
        allowedTargets: readonly RunInvocationTarget[],
    ): RunCommandPromptAvailability {
        const normalizedTargets = new Set(allowedTargets)

        return {
            bashAvailable: normalizedTargets.has("bash"),
            backgroundBashAvailable: normalizedTargets.has("background_bash"),
        }
    }

    parseInvocation(
        invocation: ExtractedRunInvocationLike,
        options?: { allowTruncated?: boolean },
    ): RunCommandParseResult | null {
        return this.getCommand(invocation.target)?.parseInvocation(invocation, options) ?? null
    }

    async executeToolCall(
        runtime: CommandRuntime,
        toolCall: ToolCall,
    ): Promise<ToolResult> {
        const command = this.resolveCommandForToolCall(toolCall)
        return command.execute(toolCall, runtime)
    }

    private resolveCommandForToolCall(toolCall: ToolCall): RunCommand {
        for (const command of this.commands) {
            if (command.matchesToolCall(toolCall)) {
                return command
            }
        }

        const toolCommand = this.getCommand("tool")
        if (!toolCommand) {
            throw new Error("Tool run command is not registered")
        }

        return toolCommand
    }
}

export const DEFAULT_RUN_COMMAND_REGISTRY = new RunCommandRegistry([
    new ToolRunCommand(),
    new BashRunCommand(),
    new BackgroundBashRunCommand(),
])

export function createCustomRunCommandRegistry(
    bashCommand: BashRunCommand,
    backgroundBashCommand: BackgroundBashRunCommand,
): RunCommandRegistry {
    return new RunCommandRegistry([
        new ToolRunCommand(),
        bashCommand,
        backgroundBashCommand,
    ])
}

export function createRunCommandRuntimeBindings(
    bashCommand: BashRunCommand,
    backgroundBashCommand: BackgroundBashRunCommand,
): {
    executeBash: (command: string) => Promise<ToolResult>
    executeBackgroundBash: (command: string) => Promise<ToolResult>
    runCommandRegistry: RunCommandRegistry
} {
    return {
        executeBash: (command: string) => bashCommand.executeRaw(command),
        executeBackgroundBash: (command: string) => backgroundBashCommand.executeRaw(command),
        runCommandRegistry: createCustomRunCommandRegistry(bashCommand, backgroundBashCommand),
    }
}

export function getRunCommandPromptAvailability(
    allowedTargets: readonly RunInvocationTarget[],
): RunCommandPromptAvailability {
    return DEFAULT_RUN_COMMAND_REGISTRY.getPromptAvailability(allowedTargets)
}

export async function executeToolCallWithRunCommands(
    runtime: CommandRuntime,
    toolCall: ToolCall,
    commandRegistry: RunCommandRegistry = DEFAULT_RUN_COMMAND_REGISTRY,
): Promise<ToolResult> {
    return commandRegistry.executeToolCall(runtime, toolCall)
}