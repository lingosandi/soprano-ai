export interface BuildMemoryCompactionPromptOptions {
    persona: string
    currentState: string
    messageBlock: string
    schema: string
    rules: string[]
}

export function buildMemoryCompactionPrompt(
    options: BuildMemoryCompactionPromptOptions,
): string {
    const {
        persona,
        currentState,
        messageBlock,
        schema,
        rules,
    } = options

    return `${persona}

Current compacted state:
${currentState || "(empty — first compaction)"}

Messages to compact:
${messageBlock}

Return a JSON object with exactly these fields:
${schema}

Rules:
${rules.map((rule) => `- ${rule}`).join("\n")}`
}
