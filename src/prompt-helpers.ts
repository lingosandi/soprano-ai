/**
 * Shared prompt building utilities for agentic loops.
 *
 * Used by Hammer CLI agent, Magic webapp agent, Monoslides, and Monospace to format
 * tool surfaces and construct core prompt templates.
 */

import type { ToolDefinition } from "./types"
import type { TruncatedToolInfo } from "./agent-machine"
import { formatToolDefinitions } from "./command-response-utils"
import {
    getRunCommandPromptAvailability,
    type RunInvocationTarget,
} from "./run-command-registry"
import {
    SHALLOW_WORKSPACE_DISCOVERY_RULE_LINE,
    SLUG_FORMAT_EXAMPLE_RULE_LINES,
    SLUG_SEPARATOR_EXAMPLE_BLOCK,
    SHARED_TOOL_USAGE_RULE,
    TOOL_CALL_SEPARATOR_RULE,
    type BashCommandsSectionOptions,
    buildBackgroundBashCommandsSection,
    buildBashCommandsSection,
    SHARED_TOOL_CALL_EXAMPLE_LINES,
    SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE,
    WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES,
} from "./tool-call-prompts"

export const DEFAULT_AGENT_FALLBACK_SYSTEM_PROMPT = "You are an AI agent."
export const ERROR_RECOVERY_RULE_LINE =
    "**ERROR RECOVERY**: If a tool returns `success=false`, inspect the error, fix the cause, and retry."
export const ROOT_CAUSES_RULE_LINE =
    "**ROOT CAUSES**: Fix root causes rather than suppressing symptoms."
export const VALIDATE_AFTER_CHANGES_RULE_LINE =
    "**VALIDATE AFTER CHANGES**: Check syntax, imports, and obvious integration errors after editing."
export const INCREMENTAL_TESTING_RULE_LINE =
    "**INCREMENTAL TESTING**: Write → test → fix before moving on. Verify deliverables actually work before finishing successfully."
export const CODE_QUALITY_RULE_LINE =
    "**CODE QUALITY**: For TypeScript, prefer `tsc --noEmit` or the repo's build/test command after meaningful edits."
export const SKILL_INVOKE_READ_RULE_LINE =
    "If the user says invoke /skill-name, treat that as asking you to read the skill-name skill with ReadSkill before proceeding."
export const TODO_LIST_FIRST_RESPONSE_RULE_LINE =
    "You MUST call the manage_todo_list tool in your very first response to plan your work before executing any other tool. Break the task into specific, actionable steps. On subsequent responses, update the todo list to reflect progress."

export interface FormatToolsSectionOptions {
    bashAvailable?: boolean
    backgroundBashAvailable?: boolean
    allowedRunTargets?: readonly RunInvocationTarget[]
    bashCommandsSectionOptions?: BashCommandsSectionOptions
}

export interface SystemPromptSections {
    basePrompt: string
    toolsHeading?: string
    toolsSection: string
    supplementalRules?: string
}

export interface SystemPromptBuildContext {
    tools: ToolDefinition[]
    toolsSectionOptions: FormatToolsSectionOptions
}

export type SystemPromptCustomizer = (
    sections: Readonly<SystemPromptSections>,
    context: Readonly<SystemPromptBuildContext>,
) => Partial<SystemPromptSections> | void

export function createAppendToolsSectionCustomizer(
    block: string,
): SystemPromptCustomizer {
    return (sections) => {
        const trimmedBlock = block.trim()
        if (!trimmedBlock) {
            return
        }

        return {
            toolsSection: [sections.toolsSection, trimmedBlock]
                .filter((section) => section.trim().length > 0)
                .join("\n\n"),
        }
    }
}

export function createToolsSectionOverrideCustomizer(
    buildToolsSection: (context: Readonly<SystemPromptBuildContext>) => string,
): SystemPromptCustomizer {
    return (_sections, context) => ({
        toolsSection: buildToolsSection(context),
    })
}

function applySystemPromptCustomizers(
    sections: SystemPromptSections,
    context: SystemPromptBuildContext,
    customizers?: readonly SystemPromptCustomizer[],
): SystemPromptSections {
    if (!customizers || customizers.length === 0) {
        return sections
    }

    return customizers.reduce<SystemPromptSections>((current, customizer) => ({
        ...current,
        ...(customizer(current, context) ?? {}),
    }), sections)
}

export function buildValidationRecoveryRuleLines(options?: {
    includeVerifiedCompletionRule?: boolean
    reservedControlHeadersRule?: string
}): string[] {
    const includeVerifiedCompletionRule =
        options?.includeVerifiedCompletionRule ?? false
    const reservedControlHeadersRule =
        options?.reservedControlHeadersRule
        ?? "Treat the structured control headers as reserved control syntax and use them only in the final executable segment."

    return [
        "If the previous response was rejected for format, do not explain the response protocol back to the system.",
        "Recover by choosing one concrete next action.",
        reservedControlHeadersRule,
        TOOL_CALL_SEPARATOR_RULE,
        ...SLUG_FORMAT_EXAMPLE_RULE_LINES,
        ...(includeVerifiedCompletionRule
            ? [
                "Do not claim success until your latest tool results or observed state show the task is complete.",
            ]
            : []),
    ]
}

// ---------------------------------------------------------------------------
// Tool descriptions
// ---------------------------------------------------------------------------

/**
 * Format tool definitions for injection into a system prompt.
 * Thin wrapper over `formatToolDefinitions` with configurable bash guidance.
 */
export function formatToolsSection(
    tools: ToolDefinition[],
    options?: FormatToolsSectionOptions,
): string {
    const commandPromptAvailability = options?.allowedRunTargets
        ? getRunCommandPromptAvailability(options.allowedRunTargets)
        : undefined
    let desc = formatToolDefinitions(tools, "unix")
    if (options?.bashAvailable ?? commandPromptAvailability?.bashAvailable) {
        const bashSection = buildBashCommandsSection(options?.bashCommandsSectionOptions)

        desc = [desc, bashSection].filter(Boolean).join("\n\n")
    }

    if (options?.backgroundBashAvailable ?? commandPromptAvailability?.backgroundBashAvailable) {
        const backgroundBashSection = buildBackgroundBashCommandsSection()

        desc = [desc, backgroundBashSection].filter(Boolean).join("\n\n")
    }

    return desc
}

export interface SkillSummaryLike {
    metadata: {
        name: string
        description: string
    }
}

export function buildSkillsSection(allSkills: SkillSummaryLike[]): string {
    if (allSkills.length === 0) {
        return ""
    }

    const skillSummaries = [...allSkills]
        .sort((left, right) =>
            left.metadata.name.localeCompare(right.metadata.name),
        )
        .map((skill) => `  • **${skill.metadata.name}**: ${skill.metadata.description}`)
        .join("\n")

    return `\n\n# AVAILABLE SKILLS\n\nSpecialized skills provide expert workflows and domain knowledge. Each skill contains detailed instructions, examples, and best practices.\n\n${skillSummaries}\n\nThese skill summaries are loaded into context automatically at the start of the run. If a skill is relevant, call ReadSkill to load the full instructions before implementing that workflow.\n\n**Skills guidance**:\n1. Review the skills above before starting implementation-heavy work\n2. If a listed skill clearly matches the task, prefer reading that skill before planning or implementing a generic approach\n3. Prefer the most specific relevant skill over a broader fallback skill when multiple skills could apply\n4. ${SKILL_INVOKE_READ_RULE_LINE}\n5. Use ReadSkill for the specific skill whose workflow you want to follow in detail before making substantive changes\n6. If a skill you read instructs you to consult another foundational skill, follow that dependency before implementation\n7. Skills are instructions, not replacements for the registered tools\n\n---\n\n`
}

export function buildSkillAwareStaticContext(options: {
    allSkills: SkillSummaryLike[]
    staticRules: string
}): string {
    return [
        buildSkillsSection(options.allSkills).trim(),
        options.staticRules.trim(),
    ].filter((section) => section.length > 0).join("\n\n")
}

export interface WorkspaceCodingStaticRulesOptions {
    rulesHeading?: string
    skillsDirectory?: string
    additionalRuleSections?: readonly WorkspaceCodingStaticRuleSection[]
}

export type WorkspaceCodingStaticBuiltInSection =
    | "workspace-safety"
    | "editing-strategy"
    | "verification"
    | "efficiency"

export type WorkspaceCodingStaticRuleSection =
    | {
        section: WorkspaceCodingStaticBuiltInSection
        ruleLines?: readonly string[]
        omitRuleLines?: readonly string[]
        heading?: never
    }
    | {
        heading: string
        ruleLines: readonly string[]
        section?: never
        omitRuleLines?: never
    }

function formatRuleLines(ruleLines: readonly string[]): string {
    return ruleLines
        .map((rule) => rule.trim())
        .filter((rule) => rule.length > 0)
        .map((rule) => `- ${rule}`)
        .join("\n")
}

interface WorkspaceCodingResolvedRuleSection {
    id: string
    heading: string
    ruleLines: string[]
}

export const PORT_CONFLICT_RULE_LINE =
    "**PORT CONFLICTS**: If a server fails with `EADDRINUSE` and `FreePort` is available, use it before retrying."

function buildDefaultWorkspaceCodingSections(skillsDirectory: string): WorkspaceCodingResolvedRuleSection[] {
    return [
        {
            id: "workspace-safety",
            heading: "Workspace Safety",
            ruleLines: [
                "**RELATIVE PATHS**: Use workspace-relative paths for file operations unless a tool explicitly requires an absolute path.",
                `**SKILLS DIRECTORY**: Skills live in \`${skillsDirectory}/\`.`,
                `If a skill references \`scripts/file.py\`, use \`${skillsDirectory}/{skill-name}/scripts/file.py\`.`,
                `If a skill references \`guide.md\`, inspect \`${skillsDirectory}/{skill-name}/guide.md\` with bash or another focused tool.`,
                "**READ BEFORE WRITE**: Inspect an existing file and nearby related files before modifying it unless you are creating a clearly new file in an empty target location.",
                "**CREATE PARENTS BEFORE CAT**: When creating a new file with shell redirection such as `cat > path/to/file <<'EOF'`, always run `mkdir -p path/to` first because `cat` and shell redirection do not create missing parent directories.",
            ],
        },
        {
            id: "editing-strategy",
            heading: "Editing Strategy",
            ruleLines: [
                "**SKILL-FIRST WORKFLOWS**: When the current task clearly matches an available skill, read that skill before making substantive edits, implementation plans, or design decisions.",
                "**FILE CHANGES VIA BASH**: Use a final \`---bash---\` control segment for creating, editing, appending, renaming, and normalizing files. Prefer scoped edits over blind overwrites.",
                "**TARGETED EDITS**: When modifying an existing file, always use narrow search-and-replace or patch-style edits that touch only the necessary lines. Preserve surrounding code, formatting, and structure.",
                "**SEARCH/REPLACE REQUIRED**: For existing files, use in-place search-and-replace commands such as `sed` or another scoped patching approach. Do not rewrite an existing file wholesale to make a fix.",
                "**NO EXISTING-FILE CAT REWRITES**: Never regenerate or overwrite an existing file with `cat > file <<'EOF'`. Reserve `cat > file <<'EOF'` for creating a new file that does not already exist.",
                "**ERROR FIXES SHOULD BE LOCAL**: When a compiler, linter, or runtime error points to part of an existing file, patch that local region instead of rewriting the file.",
                "**EXPLORE, PLAN, THEN IMPLEMENT**: For ambiguous, multi-file, or unfamiliar work: explore first, form a plan, then implement.",
                "**KEEP SCRIPTS SHORT**: Keep one-off scripts around 100 lines or less. Split larger scripts into verified steps.",
                `**CONTEXT GATHERING**: Use bash with ${WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES} to discover structure, inspect related files, and understand patterns before changing code.`,
                `**SHALLOW DISCOVERY**: ${SHALLOW_WORKSPACE_DISCOVERY_RULE_LINE}`,
                "**BASH SAFETY**: Avoid destructive broad-scope commands. Keep file operations tightly scoped.",
            ],
        },
        {
            id: "verification",
            heading: "Verification",
            ruleLines: [
                ERROR_RECOVERY_RULE_LINE,
                ROOT_CAUSES_RULE_LINE,
                VALIDATE_AFTER_CHANGES_RULE_LINE,
                INCREMENTAL_TESTING_RULE_LINE,
                CODE_QUALITY_RULE_LINE,
            ],
        },
        {
            id: "efficiency",
            heading: "Efficiency",
            ruleLines: [
                `**EFFICIENT FILE SEARCH**: Use bash with ${WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES} to locate files and inspect focused ranges instead of repeatedly dumping whole files.`,
                "**OUTPUT SIZE & TRUNCATION**: Keep each control-segment payload reasonably small. For large file generation, prefer a complete validated file or a sequence of coherent patch-sized edits. Use incremental append steps only when tooling or truncation constraints make that necessary.",
                "**MINIMIZE FILE CHURN**: Avoid rewriting entire existing files when a scoped replacement will do. Smaller in-place edits reduce regressions, token waste, and repeated repair loops.",
                "**PREFER SED-SHAPED FIXES**: When using bash to repair existing files, use `sed`-style or other search-and-replace edits instead of `cat`-based full-file regeneration.",
                "**EFFICIENCY**: Prefer one well-composed control-segment action per response instead of fragmented follow-up steps when a single bash command or tool invocation will do. Do not re-read files already in context without a reason.",
                PORT_CONFLICT_RULE_LINE,
            ],
        },
    ]
}

function applyWorkspaceCodingSectionOverrides(
    baseSections: WorkspaceCodingResolvedRuleSection[],
    additionalRuleSections: readonly WorkspaceCodingStaticRuleSection[],
): WorkspaceCodingResolvedRuleSection[] {
    const resolvedSections = baseSections.map((section) => ({
        ...section,
        ruleLines: [...section.ruleLines],
    }))

    for (const override of additionalRuleSections) {
        const trimmedRuleLines = (override.ruleLines ?? [])
            .map((rule) => rule.trim())
            .filter((rule) => rule.length > 0)
        const omittedRuleLines = new Set(
            (override.omitRuleLines ?? [])
                .map((rule) => rule.trim())
                .filter((rule) => rule.length > 0),
        )

        if (override.section) {
            const targetSection = resolvedSections.find((section) => section.id === override.section)

            if (!targetSection) {
                continue
            }

            if (omittedRuleLines.size > 0) {
                targetSection.ruleLines = targetSection.ruleLines.filter(
                    (rule) => !omittedRuleLines.has(rule),
                )
            }

            targetSection.ruleLines.push(...trimmedRuleLines)
            continue
        }

        const heading = override.heading?.trim() ?? ""

        if (!heading || trimmedRuleLines.length === 0) {
            continue
        }

        resolvedSections.push({
            id: `custom:${heading}`,
            heading,
            ruleLines: trimmedRuleLines,
        })
    }

    return resolvedSections
}

export function buildWorkspaceCodingStaticRules(
    options?: WorkspaceCodingStaticRulesOptions,
): string {
    const skillsDirectory = options?.skillsDirectory ?? ".claude/skills"
    const rulesHeading = options?.rulesHeading ?? "WORKSPACE CODING AGENT RULES"
    const additionalRuleSections = options?.additionalRuleSections ?? []
    const normalizedSections = applyWorkspaceCodingSectionOverrides(
        buildDefaultWorkspaceCodingSections(skillsDirectory),
        additionalRuleSections,
    )
        .map((section) => {
            if (section.ruleLines.length === 0) {
                return ""
            }

            return `## ${section.heading}\n\n${formatRuleLines(section.ruleLines)}`
        })
        .filter((section) => section.length > 0)
        .join("\n\n")

    const codingAgentRules = `

# ${rulesHeading}

${normalizedSections}`

    return buildCoreStaticRules() + codingAgentRules
}

export function buildToolAwareSystemPrompt(
    basePrompt: string,
    tools: ToolDefinition[],
    options?: {
        bashAvailable?: boolean
        backgroundBashAvailable?: boolean
        allowedRunTargets?: readonly RunInvocationTarget[]
        toolsSectionOptions?: Omit<FormatToolsSectionOptions, "bashAvailable" | "backgroundBashAvailable" | "allowedRunTargets">
        toolsHeading?: string
        systemPromptCustomizers?: readonly SystemPromptCustomizer[]
    },
): string {
    const commandPromptAvailability = options?.allowedRunTargets
        ? getRunCommandPromptAvailability(options.allowedRunTargets)
        : undefined
    const bashAvailable = options?.bashAvailable ?? commandPromptAvailability?.bashAvailable ?? false
    const backgroundBashAvailable = options?.backgroundBashAvailable ?? commandPromptAvailability?.backgroundBashAvailable ?? false
    const toolsSectionOptions: FormatToolsSectionOptions = {
        bashAvailable,
        backgroundBashAvailable,
        allowedRunTargets: options?.allowedRunTargets,
        ...options?.toolsSectionOptions,
    }
    const sections = applySystemPromptCustomizers({
        basePrompt,
        toolsHeading: options?.toolsHeading ?? "## Available Tools",
        toolsSection: formatToolsSection(tools, toolsSectionOptions),
    }, {
        tools,
        toolsSectionOptions,
    }, options?.systemPromptCustomizers)

    if (!sections.toolsSection.trim() && !bashAvailable && !backgroundBashAvailable) {
        return sections.basePrompt
    }

    const promptSections = [sections.basePrompt]

    if (sections.toolsSection.trim()) {
        if (tools.length > 0) {
            promptSections.push(`${sections.toolsHeading ?? "Available tools:"}\n${sections.toolsSection}`)
        } else {
            promptSections.push(sections.toolsSection)
        }
    }

    if (sections.supplementalRules?.trim()) {
        promptSections.push(sections.supplementalRules)
    }

    return promptSections.join("\n\n")
}

export function buildAgentIdentityLine(options: {
    identity?: string
    agentName?: string
    roleDescription?: string
}): string {
    const {
        identity,
        agentName,
        roleDescription,
    } = options

    const identityLine = identity
        ?? (agentName && roleDescription
            ? `You are ${agentName}, a ${roleDescription} agent.`
            : null)

    if (!identityLine) {
        throw new Error("buildAgentIdentityLine requires either identity or both agentName and roleDescription")
    }

    return identityLine
}

export function buildWebRuntimeRules(options?: {
    additionalRules?: string[]
    includeVerifiedCompletionRule?: boolean
}): string {
    const additionalRules = options?.additionalRules ?? []

    return [
        ...buildValidationRecoveryRuleLines({
            includeVerifiedCompletionRule: options?.includeVerifiedCompletionRule ?? false,
            reservedControlHeadersRule:
                "Treat the structured control headers as reserved control syntax. Use them only in the final executable segment, never inside analysis or summaries.",
        }),
        ...additionalRules,
    ].join("\n")
}

export function buildAgentSystemPrompt(options: {
    identityLine?: string
    tools: ToolDefinition[]
    supplementalRules?: string
    bashAvailable?: boolean
    backgroundBashAvailable?: boolean
    allowedRunTargets?: readonly RunInvocationTarget[]
    toolsSectionOptions?: Omit<FormatToolsSectionOptions, "bashAvailable" | "backgroundBashAvailable" | "allowedRunTargets">
    systemPromptCustomizers?: readonly SystemPromptCustomizer[]
}): string {
    const commandPromptAvailability = options.allowedRunTargets
        ? getRunCommandPromptAvailability(options.allowedRunTargets)
        : undefined
    const toolsSectionOptions: FormatToolsSectionOptions = {
        bashAvailable: options.bashAvailable ?? commandPromptAvailability?.bashAvailable ?? false,
        backgroundBashAvailable:
            options.backgroundBashAvailable
            ?? commandPromptAvailability?.backgroundBashAvailable
            ?? false,
        allowedRunTargets: options.allowedRunTargets,
        ...options.toolsSectionOptions,
    }
    const sections = applySystemPromptCustomizers({
        basePrompt: options.identityLine ?? DEFAULT_AGENT_FALLBACK_SYSTEM_PROMPT,
        toolsHeading: "# TOOLS",
        toolsSection: formatToolsSection(options.tools, toolsSectionOptions),
        supplementalRules: options.supplementalRules,
    }, {
        tools: options.tools,
        toolsSectionOptions,
    }, options.systemPromptCustomizers)

    return [
        sections.basePrompt,
        sections.toolsHeading,
        sections.toolsSection,
        sections.supplementalRules,
    ].filter((section): section is string => Boolean(section && section.trim().length > 0)).join("\n\n")
}

// ---------------------------------------------------------------------------
// Shared static rules
// ---------------------------------------------------------------------------

/**
 * Core static rules shared by all agent loops (Hammer, Magic, Monoslides, Monospace).
 *
 * Covers:
 *  - Shell-style output format specification
 *  - Terminal exit requirements
 *  - Tool call formatting
 *  - Scratchpad usage
 *
 * Hammer extends this with file-path rules, skills-first policy,
 * incremental testing, etc. Magic uses this as-is or with lighter extensions.
 *
 * Designed to be injected once via `memoryLayer.setStaticContext()`,
 * saving ~1500 tokens per action vs repeating in every system prompt.
 */
export function buildCoreStaticRules(): string {
    return `# AGENT RULES (Persistent — do not repeat)

## Output Format

Respond with normal prose first, then end every response with exactly one structured control block whose standalone slug header is on its own line and whose payload is on the following line(s):

I analyzed the current state.
${SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE}

## Control Segment Contract

- Treat the structured control headers as reserved control syntax — do not express tool instructions as braces, field blocks, arrays, XML tags, or bracket tags
- ${SHARED_TOOL_USAGE_RULE}
- Use the final control block only as the final executable block of the response
- ${TOOL_CALL_SEPARATOR_RULE}
- Never mention, quote, or explain the control headers in your analysis, summaries, or self-corrections
- Use a final ---bash--- control block whose payload is exit 0 only when finishing successfully, and a final ---bash--- control block whose payload is exit 1 only when finishing unsuccessfully. Those exit payloads are intercepted as control signals and are not executed as real shell commands.
- Prefer raw payloads over wrapping the entire command in outer quotes; use shell quotes only inside the payload when needed

## Control Segment Examples

${SHARED_TOOL_CALL_EXAMPLE_LINES.join("\n")}

${SLUG_SEPARATOR_EXAMPLE_BLOCK}

## Required Endings

- Include prose analysis before the final control block
- If continuing, include exactly one real action block as the final control block
- If finishing successfully, include exactly one final ---bash--- control block with payload exit 0 and no tool commands
- If finishing unsuccessfully, include exactly one final ---bash--- control block with payload exit 1 and no tool commands
- The final control block is ALWAYS the very last thing in the response. Nothing may follow it — no prose, no second header block, no commentary.

Continue is implicit when you omit an exit line. Never loop after the task is already complete.`
}

// ---------------------------------------------------------------------------
// Step user message builder (shared by Hammer, Magic, Monoslides, and Monospace)
// ---------------------------------------------------------------------------

export interface StepUserMessageOptions {
    /** Current action number. */
    actionCount: number
    /** Truncated tool info from the previous iteration (if any). */
    truncatedToolInfo?: TruncatedToolInfo
}

/**
 * Build the user message injected at the start of each agentic loop step.
 *
 * Handles truncation-specific continuation guidance (Write → "Use Append",
 * generic tool → "continue from where you left off") and the default
 * "Continue working" prompt.
 *
 * Shared by Hammer's UnifiedAgent, Magic's useAgent hook, and Monoslides/Monospace runtimes.
 */
export function buildStepUserMessage(opts: StepUserMessageOptions): string {
    const { truncatedToolInfo } = opts

    if (truncatedToolInfo) {
        if (truncatedToolInfo.name === "Write" && truncatedToolInfo.filePath) {
            if (truncatedToolInfo.executionSucceeded) {
                return `Your previous Write to "${truncatedToolInfo.filePath}" was truncated due to max_tokens limit. The partial content HAS been written successfully. You can see exactly where it ended in your previous message. Use Append to continue writing the remaining content from where you left off.`
            }
            return `Your previous Write to "${truncatedToolInfo.filePath}" was truncated due to max_tokens limit AND the tool execution failed. Check the error in the last tool result and retry with a valid path.`
        }
        return `Your previous ${truncatedToolInfo.name} was truncated due to max_tokens limit. Please continue from where you left off.`
    }

    return "Continue working on the task. What is your next action?"
}

/**
 * Check whether the last message in the conversation is a user error
 * message (prefixed with ⚠️), which means we should skip injecting
 * another user message to avoid double user messages.
 *
 * Shared by Hammer's UnifiedAgent, Magic's useAgent hook, and Monoslides/Monospace runtimes.
 */
export function shouldSkipStepUserMessage(
    lastRole: string | null,
    lastContent: string | null,
): boolean {
    return lastRole === "user" && (lastContent?.startsWith("⚠️") ?? false)
}

// ---------------------------------------------------------------------------
// Truncated tool info extraction (shared by Hammer, Magic, Monoslides, and Monospace)
// ---------------------------------------------------------------------------

export interface ToolCallLike {
    name: string
    parameters?: Record<string, unknown>
}

/**
 * Extract `TruncatedToolInfo` from the first tool call in a response
 * when the LLM response was truncated (finishReason === "length").
 *
 * Returns `undefined` if no tool calls are provided.
 */
export function extractTruncatedToolInfo(
    calls: ToolCallLike[],
): TruncatedToolInfo | undefined {
    if (calls.length === 0) return undefined
    const first = calls[0]
    return {
        name: first.name,
        filePath:
            (first.parameters?.path as string) ||
            (first.parameters?.file_path as string) ||
            undefined,
        executionSucceeded: false,
    }
}
