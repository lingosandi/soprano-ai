/**
 * Shared tool execution utilities for agentic loops.
 *
 * Used by both Hammer CLI agent and Magic webapp agent for
 * consistent tool result truncation and safe execution wrapping.
 */

import type { ToolCall, ToolResult } from "./types"
import { formatToolCallAsUnixCommand } from "./unix-tooling"

// ---------------------------------------------------------------------------
// Tool result truncation
// ---------------------------------------------------------------------------

/** Default maximum characters for tool result strings. */
export const MAX_TOOL_RESULT_CHARS = 30_000
const MAX_PRESENTATION_LINES = 200
const MAX_PRESENTATION_CHARS = 50_000

export interface TruncateOptions {
    /** Maximum character length (default: 30,000). */
    maxChars?: number
    /**
     * Truncation strategy:
     * - `"head-tail"`: Keep first half + last half (preserves end of large outputs).
     * - `"head-only"`: Keep the first N characters.
     */
    strategy?: "head-tail" | "head-only"
}

/**
 * Truncate an oversized tool result string to stay within LLM context limits.
 *
 * @returns The original string if within limit, otherwise a truncated version.
 */
export function truncateToolResult(
    resultStr: string,
    options?: TruncateOptions,
): string {
    const maxChars = options?.maxChars ?? MAX_TOOL_RESULT_CHARS
    const strategy = options?.strategy ?? "head-tail"

    if (resultStr.length <= maxChars) return resultStr

    if (strategy === "head-only") {
        return resultStr.substring(0, maxChars) + "\n...(truncated)"
    }

    // head-tail: keep first half and last half
    const half = Math.floor(maxChars / 2)
    const head = resultStr.substring(0, half)
    const tail = resultStr.substring(resultStr.length - half)
    const omitted = resultStr.length - maxChars
    return `${head}\n... [${omitted} chars truncated] ...\n${tail}`
}

// ---------------------------------------------------------------------------
// Tool result presentation
// ---------------------------------------------------------------------------

export function formatToolResultMessage(
    toolCall: ToolCall,
    result: ToolResult,
): string {
    const exitCode = typeof result.exit_code === "number"
        ? result.exit_code
        : result.success === true
            ? 0
            : 1
    const durationMs = typeof result.duration_ms === "number"
        ? result.duration_ms
        : 0
    const command = typeof result.command === "string" && result.command.length > 0
        ? result.command
        : formatPseudoCommand(toolCall)

    const stdout = truncatePresentationOutput(renderToolStdout(result))
    const stderr = renderToolStderr(result, exitCode)
    const metaLine = buildMetaLine(toolCall, result)
    const lines = [`$ ${command}`]

    if (stdout) {
        lines.push(stdout)
    }

    if (stderr) {
        if (!stdout && exitCode !== 0 && !stderr.includes("\n")) {
            lines.push(`[error] ${stderr}`)
        } else {
            lines.push(`[stderr]\n${stderr}`)
        }
    }

    if (!stdout && !stderr && exitCode === 0) {
        lines.push("(no output)")
    }

    if (metaLine) {
        lines.push(metaLine)
    }

    lines.push(`[exit:${exitCode} | ${formatDuration(durationMs)}]`)
    return lines.join("\n")
}

export function parseToolResultMessage(content: string): {
    success: boolean
    toolName?: string
    error?: string
    parsed?: Record<string, any>
} {
    try {
        const parsed = JSON.parse(content)
        const toolName = typeof parsed.command_name === "string"
            ? parsed.command_name
            : typeof parsed.route === "string"
                ? parsed.route
                : typeof parsed.tool_name === "string"
                    ? parsed.tool_name
                    : typeof parsed.toolName === "string"
                        ? parsed.toolName
                        : typeof parsed.tool === "string"
                            ? parsed.tool
                            : typeof parsed.name === "string"
                                ? parsed.name
                : undefined
        return {
            success: parsed.success === true,
            toolName,
            error: parsed.error,
            parsed,
        }
    } catch {
        const lines = content.split(/\r?\n/)
        const commandLine = lines[0]?.startsWith("$ ") ? lines.shift()!.slice(2) : undefined

        const stdoutLines: string[] = []
        const stderrLines: string[] = []
        let errorLine: string | undefined
        let metaLine = ""
        let exitCode = 1
        let durationMs = 0
        let inStderr = false

        for (const line of lines) {
            if (line.startsWith("[stderr]")) {
                inStderr = true
                const inline = line.slice("[stderr]".length).trim()
                if (inline) {
                    stderrLines.push(inline)
                }
                continue
            }

            if (line.startsWith("[error] ")) {
                errorLine = line.slice("[error] ".length)
                inStderr = false
                continue
            }

            if (line.startsWith("[meta] ")) {
                metaLine = line.slice("[meta] ".length)
                inStderr = false
                continue
            }

            const exitMatch = line.match(/^\[exit:(-?\d+)\s*\|\s*([^\]]+)\]$/)
            if (exitMatch) {
                exitCode = Number(exitMatch[1])
                durationMs = parseDuration(exitMatch[2])
                inStderr = false
                continue
            }

            if (inStderr) {
                stderrLines.push(line)
            } else {
                stdoutLines.push(line)
            }
        }

        const metadata = parseMetadata(metaLine)
        const stdout = stdoutLines.join("\n").trim()
        const stderr = stderrLines.join("\n").trim()
        const error = errorLine ?? (exitCode === 0 ? undefined : stderr || undefined)
        const parsed = {
            success: exitCode === 0,
            command: commandLine,
            output: stdout,
            stderr,
            error,
            exit_code: exitCode,
            duration_ms: durationMs,
            ...metadata,
        }

        return {
            success: exitCode === 0,
            toolName:
                typeof metadata.tool === "string"
                    ? metadata.tool
                    : typeof metadata.route === "string"
                        ? metadata.route
                        : undefined,
            error,
            parsed,
        }
    }
}

// ---------------------------------------------------------------------------
// Safe tool execution
// ---------------------------------------------------------------------------

/**
 * Execute a tool call with standard error handling.
 * Catches any thrown error and returns `{ success: false, error: message }`.
 */
export async function executeToolSafe(
    fn: () => Promise<ToolResult>,
): Promise<ToolResult> {
    try {
        return await fn()
    } catch (err: unknown) {
        return {
            success: false,
            error: err instanceof Error ? err.message : "Tool execution failed",
        }
    }
}

function renderToolStdout(result: ToolResult): string {
    if (typeof result.stdout === "string") {
        return result.stdout
    }

    if (typeof result.output === "string") {
        return result.output
    }

    if (typeof result.content === "string") {
        return result.content
    }

    if (typeof result.data === "string") {
        return result.data
    }

    const matchEntries = (result as ToolResult & { matches?: unknown[] }).matches
    if (Array.isArray(matchEntries)) {
        return matchEntries
            .map((match) => {
                if (typeof match === "string") return match
                if (!match || typeof match !== "object") return JSON.stringify(match)

                const typed = match as {
                    file?: unknown
                    line?: unknown
                    content?: unknown
                }
                const file = typeof typed.file === "string" ? typed.file : ""
                const line = typeof typed.line === "number" ? typed.line : ""
                const content = typeof typed.content === "string"
                    ? typed.content
                    : JSON.stringify(match)
                return `${file}${line ? `:${line}` : ""}: ${content}`.trim()
            })
            .join("\n")
    }

    for (const candidate of [result.output, result.content, result.data]) {
        if (candidate && typeof candidate === "object") {
            return JSON.stringify(candidate, null, 2)
        }
    }

    const {
        success: _success,
        error: _error,
        stderr: _stderr,
        exit_code: _exitCode,
        duration_ms: _durationMs,
        command: _command,
        ...remainder
    } = result
    if (Object.keys(remainder).length === 0) {
        return ""
    }

    return JSON.stringify(remainder, null, 2)
}

function renderToolStderr(result: ToolResult, exitCode: number): string {
    if (typeof result.stderr === "string" && result.stderr.length > 0) {
        return result.stderr
    }

    if (exitCode !== 0 && typeof result.error === "string") {
        return result.error
    }

    return ""
}

function truncatePresentationOutput(output: string): string {
    if (!output) {
        return ""
    }

    const lines = output.split(/\r?\n/)
    if (lines.length <= MAX_PRESENTATION_LINES && output.length <= MAX_PRESENTATION_CHARS) {
        return output
    }

    const truncated = lines.slice(0, MAX_PRESENTATION_LINES).join("\n")
    return `${truncated}\n--- output truncated (${lines.length} lines, ${output.length} chars) ---\nUse more specific commands, filters, or line ranges to narrow the result.`
}

function formatPseudoCommand(toolCall: ToolCall): string {
    const unixCommand = formatToolCallAsUnixCommand(toolCall)
    if (unixCommand) {
        return unixCommand
    }

    const parts = [toolCall.name]

    for (const [name, value] of Object.entries(toolCall.parameters ?? {})) {
        if (value === undefined || value === null) {
            continue
        }

        if (typeof value === "boolean") {
            parts.push(value ? `--${name}` : `--no-${name}`)
            continue
        }

        if (typeof value === "string" && !value.includes("\n") && !/\s/.test(value)) {
            parts.push(`--${name}`, value)
            continue
        }

        parts.push(`--${name}`, JSON.stringify(value))
    }

    return parts.join(" ")
}

function buildMetaLine(toolCall: ToolCall, result: ToolResult): string {
    const commandName = (result as ToolResult & { command_name?: unknown }).command_name
    const route = (result as ToolResult & { route?: unknown }).route
    const metadata: Record<string, string | number | boolean> = {
        tool: typeof commandName === "string" ? commandName : toolCall.name,
        route: typeof route === "string" ? route : toolCall.name,
    }

    for (const source of [toolCall.parameters ?? {}, result as Record<string, unknown>]) {
        for (const key of ["path", "url", "query", "taskId", "task_id", "pattern"]) {
            const value = source[key]
            if (typeof value === "string" && value.length > 0) {
                metadata[key] = value
            }
        }
    }

    const pairs = Object.entries(metadata)
    if (pairs.length === 0) {
        return ""
    }

    return `[meta] ${pairs
        .map(([key, value]) => `${key}=${typeof value === "string" ? JSON.stringify(value) : String(value)}`)
        .join(" ")}`
}

function parseMetadata(text: string): Record<string, any> {
    if (!text) {
        return {}
    }

    const values: Record<string, any> = {}
    const regex = /(\w+)=((?:"(?:[^"\\]|\\.)*")|\S+)/g
    let match: RegExpExecArray | null
    while ((match = regex.exec(text)) !== null) {
        const key = match[1]
        const rawValue = match[2]
        if (rawValue.startsWith('"')) {
            try {
                values[key] = JSON.parse(rawValue)
            } catch {
                values[key] = rawValue.slice(1, -1)
            }
            continue
        }

        if (rawValue === "true") {
            values[key] = true
        } else if (rawValue === "false") {
            values[key] = false
        } else if (!Number.isNaN(Number(rawValue))) {
            values[key] = Number(rawValue)
        } else {
            values[key] = rawValue
        }
    }

    return values
}

function formatDuration(durationMs: number): string {
    if (durationMs >= 1000) {
        return `${(durationMs / 1000).toFixed(1)}s`
    }
    return `${Math.max(0, Math.round(durationMs))}ms`
}

function parseDuration(raw: string): number {
    const trimmed = raw.trim().toLowerCase()
    if (trimmed.endsWith("ms")) {
        return Number(trimmed.slice(0, -2)) || 0
    }
    if (trimmed.endsWith("s")) {
        return Math.round((Number(trimmed.slice(0, -1)) || 0) * 1000)
    }
    return Number(trimmed) || 0
}

