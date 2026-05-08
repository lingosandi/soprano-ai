/**
 * Zod schemas for LLM response validation and normalization.
 *
 * Handles common model mistakes such as garbled status strings.
 *
 * Shared across Hammer, Magic, Monoslides, Monospace, and any other agent consumer.
 */

import { z } from "zod"

function normalizeToolCallName(name: string): string {
    return name.replace(/[\s_-]+/g, "").toLowerCase()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

export const ToolCallSchema = z
    .object({
        kind: z.enum(["tool", "bash", "background_bash"]).optional(),
        name: z.string().min(1, "Tool call name is required"),
        parameters: z.record(z.string(), z.any()).default({}),
        rawInvocation: z.string().optional(),
        truncated: z.boolean().optional(),
    })
    .strict()
    .superRefine((value, ctx) => {
        const normalizedName = normalizeToolCallName(value.name)

        if (normalizedName === "bash" && value.kind !== "bash") {
            ctx.addIssue({
                code: z.ZodIssueCode.custom,
                path: ["name"],
                message: 'Bash is not a registered tool name. Use kind: "bash" instead.',
            })
        }

        if (normalizedName === "backgroundbash" && value.kind !== "background_bash") {
            ctx.addIssue({
                code: z.ZodIssueCode.custom,
                path: ["name"],
                message: 'BackgroundBash is not a registered tool name. Use kind: "background_bash" instead.',
            })
        }
    })

export const LLMResponseSchema = z
    .object({
        content: z.string().optional(),
        reasoning: z.string().default("No reasoning provided"),
        scratchpad: z.string().optional(),
        selectedToolCall: ToolCallSchema.optional(),
        outcome: z.preprocess(
            (v) => (typeof v === "string" ? v.toLowerCase() : v),
            z.enum(["continue", "success", "failure"]),
        ),
    })
    .strict()
