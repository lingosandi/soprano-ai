/**
 * Shared structured control-segment prompt fragments used across agent loops.
 *
 * These helpers centralize the plain header-block response contract while
 * letting each agent layer domain-specific constraints on top.
 */

const TOOL_CALL_RUN_LINE_RULE =
    'End every response with exactly one final executable control block. Put prose on its own line(s), put the standalone slug header on its own line, then put the executable payload on the next line(s). Never place prose, the slug header, or the payload on the same line.'

const TOOL_CALL_SHELL_LINE_RULE =
    'Use plain prose for analysis and reserve the structured control block for the final executable block only.'

const TOOL_CALL_SINGLE_LINE_RULE =
    'Never emit multiple structured control blocks in a single response. The final control block is always the final thing in the response and nothing may follow it.'

export const TOOL_CALL_SEPARATOR_RULE =
    'The slug header is a hard separator between prose/thought and command/action. Put all prose before the slug, put the standalone slug header on its own line, and put only the executable payload on the following line(s). There must be a newline before the slug header and a newline after the slug header.'

export function formatStructuredControlSegment(
    target: "tool" | "bash" | "background_bash",
    payload: string,
): string {
    return [`---${target}---`, payload].join("\n")
}

export const CORRECT_SINGLE_SLUG_SEGMENT_EXAMPLE =
    formatStructuredControlSegment("bash", "ls -la")

export const CORRECT_NEWLINE_FINISH_EXAMPLE = [
    'Task complete.',
    formatStructuredControlSegment("bash", "exit 0"),
].join("\n")

export const CORRECT_NEWLINE_TOOL_EXAMPLE = [
    'I\'ll read the create-app skill first.',
    formatStructuredControlSegment("tool", 'ReadSkill --skill_name "create-app"'),
].join("\n")

export const INCORRECT_REPEATED_SLUG_EXAMPLE =
    '---bash--- ls -la ---bash---'

export const INCORRECT_MIXED_SLUG_EXAMPLE =
    '---bash--- ls -la ---tool--- BraveWebSearch "hello world"'

export const INCORRECT_INLINE_PROSE_AND_SLUG_EXAMPLE =
    'Task complete. ---bash--- exit 0'

export const INCORRECT_INLINE_PROSE_AND_TOOL_SLUG_EXAMPLE =
    'I\'ll build a landing page for Monako Glass smart glasses. Let me first read the create-app skill to ensure I scaffold this project correctly. ---tool--- ReadSkill --skill_name "create-app"'

export const INCORRECT_INLINE_HEADER_AND_PAYLOAD_EXAMPLE =
    '---tool--- ReadSkill --skill_name "create-app"'

export const SLUG_FORMAT_EXAMPLE_RULE_LINES = [
    'Correct example: prose first, then one slug header, then one executable payload block.',
    `Correct example:\n${CORRECT_SINGLE_SLUG_SEGMENT_EXAMPLE}`,
    'Correct finish example: keep the prose, slug header, and payload on separate lines.',
    `Correct finish example:\n${CORRECT_NEWLINE_FINISH_EXAMPLE}`,
    'Correct tool example: keep prose, the ---tool--- header, and the tool payload on separate lines.',
    `Correct tool example:\n${CORRECT_NEWLINE_TOOL_EXAMPLE}`,
    'Incorrect example: do not repeat the same slug in one response.',
    `Incorrect example:\n${INCORRECT_REPEATED_SLUG_EXAMPLE}`,
    'Incorrect example: do not mix different slug headers in one response.',
    `Incorrect example:\n${INCORRECT_MIXED_SLUG_EXAMPLE}`,
    'Incorrect example: do not keep prose, the slug header, and the payload on one line.',
    `Incorrect example:\n${INCORRECT_INLINE_PROSE_AND_SLUG_EXAMPLE}`,
    'Incorrect example: do not keep prose and a ---tool--- invocation on one line.',
    `Incorrect example:\n${INCORRECT_INLINE_PROSE_AND_TOOL_SLUG_EXAMPLE}`,
    'Incorrect example: do not put a slug header and its payload on the same line.',
    `Incorrect example:\n${INCORRECT_INLINE_HEADER_AND_PAYLOAD_EXAMPLE}`,
]

export const CORRECT_SEPARATOR_RESPONSE_EXAMPLE = [
    'I analyzed the current state.',
    CORRECT_SINGLE_SLUG_SEGMENT_EXAMPLE,
].join("\n")

export const SLUG_SEPARATOR_EXAMPLE_BLOCK = [
    'Correct separator example:',
    CORRECT_SEPARATOR_RESPONSE_EXAMPLE,
    '',
    'Correct finish example:',
    CORRECT_NEWLINE_FINISH_EXAMPLE,
    '',
    'Correct tool example:',
    CORRECT_NEWLINE_TOOL_EXAMPLE,
    '',
    'Incorrect multi-slug examples:',
    INCORRECT_REPEATED_SLUG_EXAMPLE,
    INCORRECT_MIXED_SLUG_EXAMPLE,
    '',
    'Incorrect inline prose-plus-slug example:',
    INCORRECT_INLINE_PROSE_AND_SLUG_EXAMPLE,
    '',
    'Incorrect inline prose-plus-tool example:',
    INCORRECT_INLINE_PROSE_AND_TOOL_SLUG_EXAMPLE,
    '',
    'Incorrect inline slug-plus-payload example:',
    INCORRECT_INLINE_HEADER_AND_PAYLOAD_EXAMPLE,
].join("\n")

export const SHARED_TOOL_USAGE_RULE =
    "Use only the tools that are actually listed in the tool section. Do not invent tools, parameters, wrapper flags, bare assignment-style arguments, or capabilities that are not explicitly listed. Follow each tool's description, usage surface, and parameter schema exactly."

export const FOLLOW_TOOL_USAGE_SURFACE_RULE =
    "For registered tools, follow the exact CLI usage shown in the tool section."

export const POSITIONAL_ARGUMENT_USAGE_RULE =
    "Use positional arguments only when the usage shows <param>; otherwise use --flag value."

export const BARE_ASSIGNMENT_STYLE_ARGUMENT_RULE =
    "Never write bare assignment-style arguments like skill_name=..., query=..., or path=... after a tool name unless that exact syntax is explicitly shown in the tool section."

export const PASSTHROUGH_TOOL_ARGUMENT_RULE =
    "Passthrough tools take their raw arguments directly after the tool name; do not invent wrapper flags unless the tool description explicitly documents them."

export const UNIX_TOOL_USAGE_GUIDANCE_LINE =
    'Use the exact registered tool name and usage shown: <param> is positional, otherwise use --flag value. Never switch to bare assignment-style arguments unless the tool explicitly shows them.'

export const UNIX_PASSTHROUGH_TOOL_GUIDANCE_LINE =
    'For passthrough tools, pass raw arguments directly after the tool name; do not invent wrapper flags unless the tool explicitly documents them.'

export const WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES =
    "rg, sed, find, ls, grep, cat"

export const SHALLOW_WORKSPACE_DISCOVERY_RULE_LINE =
    'When exploring the workspace, avoid recursive whole-tree listings such as `ls -R` or `find . -type f` at the repo root. Prefer `rg --files`, shallow `find`/`ls`, or targeted directory inspection, and exclude heavy directories such as `node_modules`, `.git`, `.next`, `dist`, `build`, and coverage outputs unless the task specifically requires them.'

export interface ShellNativeWorkflowCommandExamplesOptions {
    includeGit?: boolean
    includeBun?: boolean
    includeCurl?: boolean
}

export function buildShellNativeWorkflowCommandExamples(
    options?: ShellNativeWorkflowCommandExamplesOptions,
): string {
    const commands = [WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES]

    if (options?.includeGit ?? true) {
        commands.push("git")
    }

    if (options?.includeBun ?? true) {
        commands.push("bun")
    }

    if (options?.includeCurl ?? true) {
        commands.push("curl")
    }

    return commands.join(", ")
}

export const SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES =
    buildShellNativeWorkflowCommandExamples()

export const JUST_BASH_SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES =
    buildShellNativeWorkflowCommandExamples({
        includeGit: false,
        includeBun: false,
        includeCurl: false,
    })

export const JUST_BASH_SCRIPT_EXECUTION_RESTRICTION_LINES = [
    'Never try to run scripts with `python`, `python3`, `node`, `nodejs`, `bun`, or similar runtimes unless a listed tool explicitly provides that capability.',
]

export interface BashCommandsSectionOptions {
    shellNativeWorkflowCommandExamples?: string
    additionalGuidanceLines?: string[]
}

function buildBashCommandsSectionLines(options?: BashCommandsSectionOptions): string[] {
    const shellNativeWorkflowCommandExamples =
        options?.shellNativeWorkflowCommandExamples
        ?? SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES
    const additionalGuidanceLines = (options?.additionalGuidanceLines ?? [])
        .map((line) => line.trim())
        .filter((line) => line.length > 0)

    return [
        '## Bash Commands',
        '',
        '- Invoke them with a standalone `---bash---` header followed by the bash payload on the next line(s).',
        `- Use bash for shell-native workflows such as ${shellNativeWorkflowCommandExamples}.`,
        `- ${SHALLOW_WORKSPACE_DISCOVERY_RULE_LINE}`,
        '- Prefer registered tools when a listed tool already covers the task. Use bash for shell-native workflows or when no listed tool matches.',
        '- When fixing or updating an existing file, always use targeted in-place edits with `sed` or another scoped search-and-replace command.',
        '- Use `cat > file <<\'EOF\'` only for creating a new file that does not already exist. Never use it to repair, revise, or overwrite an existing file.',
        `- ${BASH_NOT_REGISTERED_TOOL_RULE}`,
        ...additionalGuidanceLines.map((line) => `- ${line}`),
        `- Example:\n${formatStructuredControlSegment("bash", "sed -n '1,120p' README.md")}`,
    ]
}

const REGISTERED_TOOLS_VS_BASH_RULE =
    'Use ---tool--- for listed tools, ---bash--- for one-shot shell-native workflows, and ---background_bash--- only when a Background bash commands section is present for detached long-running shell commands.'

const BASH_NOT_REGISTERED_TOOL_RULE =
    'Bash is not a registered tool name. Never start a tool payload with Bash --command ...; use the ---bash--- header instead.'

const BACKGROUND_BASH_NOT_REGISTERED_TOOL_RULE =
    'BackgroundBash is not a registered tool name. Never start a tool payload with BackgroundBash ...; use the ---background_bash--- header instead.'

const REGISTERED_TOOL_USAGE_SURFACE_RULE =
    `Tool payloads must start with the exact registered tool name after the ---tool--- header. ${FOLLOW_TOOL_USAGE_SURFACE_RULE} ${POSITIONAL_ARGUMENT_USAGE_RULE} ${BARE_ASSIGNMENT_STYLE_ARGUMENT_RULE}`

const BASH_WORKFLOW_SELECTION_RULE =
    `Use bash for shell-native workflows such as ${SHELL_NATIVE_WORKFLOW_COMMAND_EXAMPLES} when bash is listed. Prefer the appropriate listed tool over bash when a listed tool already covers the task. For edits to existing files, always use targeted search-and-replace commands and never use full-file cat rewrites. If no listed tool covers a direct HTTP request and bash is available, use bash curl.`

const BACKGROUND_BASH_WORKFLOW_SELECTION_RULE =
    'Use background_bash only when a background bash section is listed. It starts or manages detached shell commands for blocking operations such as starting servers, preview apps, and long-running watchers. Prefer plain ---bash--- for short synchronous commands.'

const SINGLE_WELL_COMPOSED_ACTION_RULE =
    'Prefer one well-composed action over multiple unrelated tool calls.'

const TOOL_RESULT_FEEDBACK_RULE =
    'Tool results include stderr and an [exit:N | duration] footer. Use that feedback instead of guessing.'

const TOOL_AND_BASH_ROUTING_RULE =
    `${REGISTERED_TOOLS_VS_BASH_RULE} ${BASH_NOT_REGISTERED_TOOL_RULE} ${BACKGROUND_BASH_NOT_REGISTERED_TOOL_RULE} ${BASH_WORKFLOW_SELECTION_RULE} ${BACKGROUND_BASH_WORKFLOW_SELECTION_RULE}`

const TOOL_EXECUTION_STRATEGY_RULE =
    `${SINGLE_WELL_COMPOSED_ACTION_RULE} ${TOOL_RESULT_FEEDBACK_RULE}`

export const STANDARD_TOOL_CALL_FORMAT_RULES = [
    TOOL_CALL_RUN_LINE_RULE,
    TOOL_CALL_SHELL_LINE_RULE,
    TOOL_CALL_SINGLE_LINE_RULE,
    TOOL_CALL_SEPARATOR_RULE,
]

const VALIDATION_FIX_REFERENCE_REQUIRED_FORMAT =
    'Write normal prose and end with exactly one final executable control block. Put prose on its own line(s), put the standalone slug header on its own line, and put the payload on the following line(s). Do not mention the control headers anywhere else.'

const VALIDATION_FIX_REFERENCE_RULE_LINES = [
    'If continuing, end with exactly one real action block.',
    'If finishing, end with exactly one final ---bash--- control block whose payload is either exit 0 or exit 1.',
    'Do not mix finish lines with tool commands.',
    'Do not mention, quote, or explain the control headers in analysis or summaries.',
    ...STANDARD_TOOL_CALL_FORMAT_RULES,
    ...SLUG_FORMAT_EXAMPLE_RULE_LINES,
]

export const VALIDATION_FIX_REFERENCE = buildValidationFixReference(
    VALIDATION_FIX_REFERENCE_REQUIRED_FORMAT,
    VALIDATION_FIX_REFERENCE_RULE_LINES,
)

export function buildStructuredControlValidationError(details: string): string {
    return `VALIDATION_ERROR: ${details}\n\n${VALIDATION_FIX_REFERENCE}`
}

export function buildMultipleStructuredControlSegmentsValidationError(
    segmentCount: number,
): string {
    return buildStructuredControlValidationError(
        `You emitted ${segmentCount} structured control blocks in one response. Emit exactly one standalone slug header per response. The slug header is the separator between prose/thought and command/action.`,
    )
}

export const VOICE_TOOL_USAGE_RULE_LINES = [
    TOOL_AND_BASH_ROUTING_RULE,
    REGISTERED_TOOL_USAGE_SURFACE_RULE,
    TOOL_EXECUTION_STRATEGY_RULE,
]

export const VOICE_TOOL_FORMAT_RULE_LINES = [
    'The final executable control block must use one standalone header: ---tool---, ---bash---, or ---background_bash---. Put prose first, then a newline, then the standalone header on its own line, then a newline, then the payload. The header and payload must be the LAST thing in your message - never put speech after them.',
    ...STANDARD_TOOL_CALL_FORMAT_RULES,
    ...SLUG_FORMAT_EXAMPLE_RULE_LINES,
    'Only call tools that are explicitly listed below. If no tools are listed and bash is unavailable, do not emit a control block.',
]

export const VOICE_WORKSPACE_VERIFICATION_RULE_LINES = [
    "You have ZERO knowledge of the user's workspace. The ONLY way to know what files exist or contain is to use tools.",
    `If the user asks about ANY file, document, or workspace content: you MUST inspect it with a listed tool or with bash commands like ${WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES} BEFORE saying anything about it. NEVER summarize, describe, or reference file contents without actually checking them first.`,
    `If the user asks you to read, open, summarize, or explain a file: inspect it with a real tool call such as:\n${formatStructuredControlSegment("bash", "cat path/to/file")}\nor with a listed reading tool. Do NOT make up what the file says.`,
    `If the user mentions a file vaguely (e.g. "read the analysis", "open the summary") and you don't know the exact filename: use bash with ${WORKSPACE_CONTEXT_GATHERING_COMMAND_EXAMPLES} first to locate likely candidates, then inspect the best match. NEVER ask the user to clarify the filename — just look it up yourself.`,
    SHALLOW_WORKSPACE_DISCOVERY_RULE_LINE,
    'If the user asks about facts, current events, or anything you are uncertain about: verify with an appropriate listed tool, or use bash with a trusted source when bash is available. Do NOT guess.',
    'If you cannot verify something with tools, say "let me check" and use a tool — or honestly say you do not know.',
    'Violating this rule causes real harm. The user trusts your answers. Never fabricate.',
]

export const BASH_COMMANDS_SECTION_LINES = buildBashCommandsSectionLines()

export const BACKGROUND_BASH_START_EXAMPLE_LINE =
    `- Start:\n${formatStructuredControlSegment("background_bash", 'start <name> --command "<long-running server command>"')}`

function buildBackgroundBashCommandsSectionLines(): string[] {
    return [
        '## Background Bash Commands',
        '',
        '- Invoke them with a standalone `---background_bash---` header followed by the payload on the next line(s).',
        '- Use background_bash to start or manage detached shell commands without blocking the agent loop.',
        '- Use it for blocking operations such as starting servers, preview apps, or long-running watchers. Prefer `---bash---` for short synchronous commands.',
        `- ${BACKGROUND_BASH_NOT_REGISTERED_TOOL_RULE}`,
        BACKGROUND_BASH_START_EXAMPLE_LINE,
        `- Status:\n${formatStructuredControlSegment("background_bash", "status <name>")}`,
        `- Logs:\n${formatStructuredControlSegment("background_bash", "logs <name> [--tail-bytes 4000]")}`,
        `- Stop:\n${formatStructuredControlSegment("background_bash", "stop <name>")}`,
    ]
}

export const BACKGROUND_BASH_COMMANDS_SECTION_LINES = buildBackgroundBashCommandsSectionLines()

export const SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE =
    formatStructuredControlSegment("tool", 'ExactToolName --required-flag "value"')

export const INVALID_TOOL_NAME_RUN_LINE_EXAMPLE =
    formatStructuredControlSegment("tool", 'exacttoolname required-arg')

export const INVALID_ASSIGNMENT_STYLE_RUN_LINE_EXAMPLE =
    formatStructuredControlSegment("tool", 'ExactToolName required_flag="value"')

export const STANDARD_AGENT_RESPONSE_EXAMPLE = [
    "This is some useful information.",
    SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE,
].join("\n")

export const SHARED_TOOL_CALL_EXAMPLE_LINES = [
    `    ✓ ${SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE}`,
    `    ✗ ${INVALID_TOOL_NAME_RUN_LINE_EXAMPLE}`,
    `    ✗ ${INVALID_ASSIGNMENT_STYLE_RUN_LINE_EXAMPLE}`,
    `    ✓ ${STANDARD_AGENT_RESPONSE_EXAMPLE}`,
    `    ✓ ${CORRECT_NEWLINE_TOOL_EXAMPLE}`,
    `    ✗ ${INCORRECT_REPEATED_SLUG_EXAMPLE}`,
    `    ✗ ${INCORRECT_MIXED_SLUG_EXAMPLE}`,
    `    ✗ ${INCORRECT_INLINE_PROSE_AND_TOOL_SLUG_EXAMPLE}`,
    `    ✗ ${INCORRECT_INLINE_HEADER_AND_PAYLOAD_EXAMPLE}`,
]

export const CONTINUE_TOOL_CALL_RESPONSE_EXAMPLE = [
    "I need to inspect the file first.",
    SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE,
].join("\n")

export const EXIT_SUCCESS_RESPONSE_EXAMPLE = [
    "Everything is verified.",
    formatStructuredControlSegment("bash", "exit 0"),
].join("\n")

export const EXIT_FAILURE_RESPONSE_EXAMPLE = [
    "This cannot be completed with the available tools.",
    formatStructuredControlSegment("bash", "exit 1"),
].join("\n")

export const VOICE_TOOL_CALL_RESPONSE_EXAMPLE = [
    "Let me check that.",
    SINGLE_TOOL_CALL_RUN_LINE_EXAMPLE,
].join("\n")

export function buildBashCommandsSection(options?: BashCommandsSectionOptions): string {
    return buildBashCommandsSectionLines(options).join("\n")
}

export function buildBackgroundBashCommandsSection(
): string {
    return buildBackgroundBashCommandsSectionLines().join("\n")
}

export function buildVoiceToolUsagePrompt(): string {
    return [
        "When you do need a tool, first speak your reasoning or acknowledgement naturally, then insert a newline and end your message with exactly one structured control block whose header is on its own line and whose payload is on the following line(s):",
        VOICE_TOOL_CALL_RESPONSE_EXAMPLE,
        "",
        formatPromptRules(VOICE_TOOL_USAGE_RULE_LINES),
    ].join("\n")
}

export function buildValidationFixReference(
    requiredFormat: string,
    rules: string[],
): string {
    return `Fix your response and retry.\n${requiredFormat}\n${formatPromptRules(rules)}`
}

export function formatPromptRules(rules: string[]): string {
    return rules.map((rule) => `- ${rule}`).join("\n")
}