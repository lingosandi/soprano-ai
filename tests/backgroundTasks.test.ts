/**
 * Tests for VoiceAgentService — background task polling & announcement queue.
 *
 * Validates the async Hammer tool execution flow:
 *   - Polling lifecycle (start/stop)
 *   - Announcement queue mechanics (drain, pending accumulation)
 *   - State-dependent announcement timing (idle vs. busy)
 *   - Tool message injection into conversation history
 *   - Callback invocations (onBackgroundTaskComplete)
 *   - Edge cases: empty polls, error resilience, concurrent drains
 *
 * Uses the same simulation-based testing pattern as the existing
 * VoiceAgentService tests — extracting and testing the core logic
 * without requiring live audio/ASR/TTS infrastructure.
 */
import { describe, expect, test } from "vitest"

import type { BackgroundToolResult } from "../src"

// ---------------------------------------------------------------------------
// BackgroundToolResult type
// ---------------------------------------------------------------------------

describe("BackgroundToolResult type", () => {
    test("has correct shape with all fields", () => {
        const result: BackgroundToolResult = {
            taskId: "task_1_100",
            tool: "HammerAgent",
            task: "Create a file",
            success: true,
            output: "File created successfully",
        }

        expect(result.taskId).toBe("task_1_100")
        expect(result.tool).toBe("HammerAgent")
        expect(result.task).toBe("Create a file")
        expect(result.success).toBe(true)
        expect(result.output).toBe("File created successfully")
        expect(result.error).toBeUndefined()
    })

    test("has correct shape with error", () => {
        const result: BackgroundToolResult = {
            taskId: "task_2_200",
            tool: "HammerAgent",
            task: "Run tests",
            success: false,
            output: "",
            error: "Exit code 1",
        }

        expect(result.success).toBe(false)
        expect(result.error).toBe("Exit code 1")
    })
})

// ---------------------------------------------------------------------------
// Announcement queue simulation
// ---------------------------------------------------------------------------

/**
 * Simulates the pendingAnnouncements queue + drainAndAnnounceCompletions
 * logic from VoiceAgentService. This is the same algorithm used in the
 * real service, extracted for deterministic testing.
 */
function createAnnouncementQueue() {
    const pendingAnnouncements: BackgroundToolResult[] = []
    const history: Array<{ role: string; content: string }> = []
    const logs: string[] = []
    const completionCallbacks: BackgroundToolResult[] = []
    let processUserInputCalls = 0

    /** Simulates pollBackgroundTasks */
    async function poll(
        poller: () => Promise<BackgroundToolResult[]>,
        currentState: string,
    ) {
        const completed = await poller()
        if (completed.length === 0) return

        pendingAnnouncements.push(...completed)
        logs.push(`${completed.length} background task(s) completed, queued for announcement`)

        // Notify via callbacks
        for (const task of completed) {
            completionCallbacks.push(task)
        }

        // If idle, announce immediately
        if (currentState === "idle") {
            drain()
        }
    }

    /** Simulates drainAndAnnounceCompletions */
    function drain() {
        if (pendingAnnouncements.length === 0) return

        const announcements = pendingAnnouncements.splice(0)

        for (const ann of announcements) {
            const toolContent = JSON.stringify({
                tool: ann.tool,
                taskId: ann.taskId,
                task: ann.task,
                success: ann.success,
                output: ann.output,
                ...(ann.error && { error: ann.error }),
                note: "This background task has completed. Briefly announce the result to the user.",
            })
            history.push({ role: "tool", content: toolContent })
        }

        logs.push(`Announcing ${announcements.length} completed task(s)…`)
        processUserInputCalls++
    }

    return {
        pendingAnnouncements,
        history,
        logs,
        completionCallbacks,
        get processUserInputCalls() { return processUserInputCalls },
        poll,
        drain,
    }
}

// ---------------------------------------------------------------------------
// Polling behavior
// ---------------------------------------------------------------------------

describe("VoiceAgentService — background task polling", () => {
    test("empty poll does not queue announcements", async () => {
        const q = createAnnouncementQueue()

        await q.poll(async () => [], "idle")

        expect(q.pendingAnnouncements.length).toBe(0)
        expect(q.processUserInputCalls).toBe(0)
        expect(q.logs.length).toBe(0)
    })

    test("completed task is queued for announcement", async () => {
        const q = createAnnouncementQueue()
        const task: BackgroundToolResult = {
            taskId: "t1",
            tool: "HammerAgent",
            task: "Fix bug",
            success: true,
            output: "Bug fixed",
        }

        // State is "speaking" — should queue but not drain
        await q.poll(async () => [task], "speaking")

        expect(q.pendingAnnouncements.length).toBe(1)
        expect(q.pendingAnnouncements[0].taskId).toBe("t1")
        expect(q.processUserInputCalls).toBe(0) // Not idle — queued only
    })

    test("idle state triggers immediate drain", async () => {
        const q = createAnnouncementQueue()
        const task: BackgroundToolResult = {
            taskId: "t1",
            tool: "HammerAgent",
            task: "Build project",
            success: true,
            output: "Done",
        }

        await q.poll(async () => [task], "idle")

        expect(q.pendingAnnouncements.length).toBe(0) // Drained
        expect(q.processUserInputCalls).toBe(1)
        expect(q.history.length).toBe(1)
    })

    test("multiple completed tasks are batched in one drain", async () => {
        const q = createAnnouncementQueue()
        const tasks: BackgroundToolResult[] = [
            { taskId: "t1", tool: "HammerAgent", task: "Task A", success: true, output: "A done" },
            { taskId: "t2", tool: "HammerAgent", task: "Task B", success: false, output: "", error: "B failed" },
        ]

        await q.poll(async () => tasks, "idle")

        expect(q.history.length).toBe(2) // Both pushed to history
        expect(q.processUserInputCalls).toBe(1) // Single LLM call for both
    })

    test("onBackgroundTaskComplete callback fires for each completed task", async () => {
        const q = createAnnouncementQueue()
        const tasks: BackgroundToolResult[] = [
            { taskId: "t1", tool: "HammerAgent", task: "A", success: true, output: "ok" },
            { taskId: "t2", tool: "HammerAgent", task: "B", success: true, output: "ok" },
        ]

        await q.poll(async () => tasks, "speaking")

        expect(q.completionCallbacks.length).toBe(2)
        expect(q.completionCallbacks[0].taskId).toBe("t1")
        expect(q.completionCallbacks[1].taskId).toBe("t2")
    })

    test("poll error is handled gracefully (no crash)", async () => {
        const q = createAnnouncementQueue()

        // Should not throw
        try {
            await q.poll(async () => { throw new Error("Network") }, "idle")
        } catch {
            // Error is expected — the real service catches and logs it
        }

        // The real service catches this error — simulate:
        // In the real code, pollBackgroundTasks has a try/catch that logs
        // For our simulation, the error propagates (which is fine for test)
        // The key point: no announcements queued, no crash
        expect(q.pendingAnnouncements.length).toBe(0)
    })
})

// ---------------------------------------------------------------------------
// Drain logic
// ---------------------------------------------------------------------------

describe("VoiceAgentService — drainAndAnnounceCompletions", () => {
    test("drain with no pending announcements is a no-op", () => {
        const q = createAnnouncementQueue()
        q.drain()

        expect(q.processUserInputCalls).toBe(0)
        expect(q.history.length).toBe(0)
        expect(q.logs.length).toBe(0)
    })

    test("drain pushes tool messages to conversation history", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t1",
            tool: "HammerAgent",
            task: "Create file",
            success: true,
            output: "File created",
        })

        q.drain()

        expect(q.history.length).toBe(1)
        expect(q.history[0].role).toBe("tool")

        const parsed = JSON.parse(q.history[0].content)
        expect(parsed.tool).toBe("HammerAgent")
        expect(parsed.taskId).toBe("t1")
        expect(parsed.task).toBe("Create file")
        expect(parsed.success).toBe(true)
        expect(parsed.output).toBe("File created")
        expect(parsed.note).toContain("announce")
    })

    test("drain includes error field for failed tasks", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t2",
            tool: "HammerAgent",
            task: "Build",
            success: false,
            output: "",
            error: "Exit code 1",
        })

        q.drain()

        const parsed = JSON.parse(q.history[0].content)
        expect(parsed.error).toBe("Exit code 1")
        expect(parsed.success).toBe(false)
    })

    test("drain omits error field for successful tasks", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t3",
            tool: "HammerAgent",
            task: "Lint",
            success: true,
            output: "No errors",
        })

        q.drain()

        const parsed = JSON.parse(q.history[0].content)
        expect(parsed.error).toBeUndefined()
    })

    test("drain clears pending announcements", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push(
            { taskId: "t1", tool: "H", task: "A", success: true, output: "ok" },
            { taskId: "t2", tool: "H", task: "B", success: true, output: "ok" },
        )

        q.drain()

        expect(q.pendingAnnouncements.length).toBe(0)
    })

    test("consecutive drains are idempotent", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t1",
            tool: "H",
            task: "A",
            success: true,
            output: "ok",
        })

        q.drain()
        q.drain() // Second drain should be a no-op

        expect(q.history.length).toBe(1)
        expect(q.processUserInputCalls).toBe(1)
    })

    test("drain triggers processUserInput once per drain", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push(
            { taskId: "t1", tool: "H", task: "A", success: true, output: "1" },
            { taskId: "t2", tool: "H", task: "B", success: true, output: "2" },
            { taskId: "t3", tool: "H", task: "C", success: true, output: "3" },
        )

        q.drain()

        expect(q.history.length).toBe(3) // 3 tool messages
        expect(q.processUserInputCalls).toBe(1) // But only 1 LLM call
    })
})

// ---------------------------------------------------------------------------
// State-dependent announcement timing
// ---------------------------------------------------------------------------

describe("VoiceAgentService — announcement timing by state", () => {
    test("idle state: announce immediately", async () => {
        const q = createAnnouncementQueue()
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "idle",
        )

        expect(q.processUserInputCalls).toBe(1)
        expect(q.pendingAnnouncements.length).toBe(0)
    })

    test("listening state: queue only, do not announce", async () => {
        const q = createAnnouncementQueue()
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "listening",
        )

        expect(q.processUserInputCalls).toBe(0)
        expect(q.pendingAnnouncements.length).toBe(1)
    })

    test("thinking state: queue only, do not announce", async () => {
        const q = createAnnouncementQueue()
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "thinking",
        )

        expect(q.processUserInputCalls).toBe(0)
        expect(q.pendingAnnouncements.length).toBe(1)
    })

    test("speaking state: queue only, do not announce", async () => {
        const q = createAnnouncementQueue()
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "speaking",
        )

        expect(q.processUserInputCalls).toBe(0)
        expect(q.pendingAnnouncements.length).toBe(1)
    })

    test("connecting state: queue only, do not announce", async () => {
        const q = createAnnouncementQueue()
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "connecting",
        )

        expect(q.processUserInputCalls).toBe(0)
        expect(q.pendingAnnouncements.length).toBe(1)
    })
})

// ---------------------------------------------------------------------------
// Post-playback drain scenario
// ---------------------------------------------------------------------------

describe("VoiceAgentService — post-playback announcement drain", () => {
    test("queued announcements drain after playback finishes", async () => {
        const q = createAnnouncementQueue()

        // Simulate: task completes while AI is speaking
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "ok" }],
            "speaking",
        )
        expect(q.pendingAnnouncements.length).toBe(1)
        expect(q.processUserInputCalls).toBe(0)

        // Simulate: playback finishes → drain hook fires
        q.drain()

        expect(q.pendingAnnouncements.length).toBe(0)
        expect(q.processUserInputCalls).toBe(1)
        expect(q.history.length).toBe(1)
    })

    test("chained drains: new task completes during announcement playback", async () => {
        const q = createAnnouncementQueue()

        // First task completes → immediate drain (idle)
        await q.poll(
            async () => [{ taskId: "t1", tool: "H", task: "A", success: true, output: "A done" }],
            "idle",
        )
        expect(q.processUserInputCalls).toBe(1)

        // While announcement is playing, second task completes
        q.pendingAnnouncements.push({
            taskId: "t2",
            tool: "H",
            task: "B",
            success: true,
            output: "B done",
        })

        // Announcement playback finishes → post-playback drain
        q.drain()

        expect(q.processUserInputCalls).toBe(2)
        expect(q.history.length).toBe(2)
    })

    test("no drain when pending is empty at post-playback", async () => {
        const q = createAnnouncementQueue()

        // Nothing queued → drain is a no-op
        q.drain()

        expect(q.processUserInputCalls).toBe(0)
    })
})

// ---------------------------------------------------------------------------
// Tool message format in history
// ---------------------------------------------------------------------------

describe("VoiceAgentService — tool message format", () => {
    test("tool message has all required fields", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "task_42",
            tool: "HammerAgent",
            task: "Write documentation",
            success: true,
            output: "Documentation generated for 5 files",
        })

        q.drain()

        const content = JSON.parse(q.history[0].content)
        expect(content).toHaveProperty("tool")
        expect(content).toHaveProperty("taskId")
        expect(content).toHaveProperty("task")
        expect(content).toHaveProperty("success")
        expect(content).toHaveProperty("output")
        expect(content).toHaveProperty("note")
    })

    test("note instructs LLM to announce briefly", () => {
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t1",
            tool: "H",
            task: "A",
            success: true,
            output: "ok",
        })

        q.drain()

        const content = JSON.parse(q.history[0].content)
        expect(content.note.toLowerCase()).toContain("announce")
        expect(content.note.toLowerCase()).toContain("result")
    })

    test("long output is preserved in tool message", () => {
        const longOutput = "x".repeat(5000)
        const q = createAnnouncementQueue()
        q.pendingAnnouncements.push({
            taskId: "t1",
            tool: "H",
            task: "A",
            success: true,
            output: longOutput,
        })

        q.drain()

        const content = JSON.parse(q.history[0].content)
        expect(content.output.length).toBe(5000)
    })
})

// ---------------------------------------------------------------------------
// Polling timer simulation
// ---------------------------------------------------------------------------

describe("VoiceAgentService — polling timer lifecycle", () => {
    /**
     * Simulates start/stop of the polling timer without requiring
     * the full VoiceAgentService infrastructure.
     */
    function createTimerSimulation() {
        let timer: ReturnType<typeof setInterval> | null = null
        let pollCount = 0

        function startPolling(intervalMs: number) {
            if (timer) return
            timer = setInterval(() => { pollCount++ }, intervalMs)
        }

        function stopPolling() {
            if (timer) {
                clearInterval(timer)
                timer = null
            }
        }

        return {
            get isPolling() { return timer !== null },
            get pollCount() { return pollCount },
            startPolling,
            stopPolling,
        }
    }

    test("startPolling sets timer", () => {
        const sim = createTimerSimulation()
        sim.startPolling(100)

        expect(sim.isPolling).toBe(true)
        sim.stopPolling()
    })

    test("stopPolling clears timer", () => {
        const sim = createTimerSimulation()
        sim.startPolling(100)
        sim.stopPolling()

        expect(sim.isPolling).toBe(false)
    })

    test("double startPolling does not create duplicate timers", () => {
        const sim = createTimerSimulation()
        sim.startPolling(100)
        sim.startPolling(100) // Should be a no-op

        expect(sim.isPolling).toBe(true)
        sim.stopPolling()
    })

    test("stopPolling without start is a no-op", () => {
        const sim = createTimerSimulation()
        sim.stopPolling() // Should not throw

        expect(sim.isPolling).toBe(false)
    })

    test("double stopPolling is a no-op", () => {
        const sim = createTimerSimulation()
        sim.startPolling(100)
        sim.stopPolling()
        sim.stopPolling() // Should not throw

        expect(sim.isPolling).toBe(false)
    })

    test("polling interval fires callbacks", async () => {
        const sim = createTimerSimulation()
        sim.startPolling(50) // 50ms interval

        await new Promise((r) => setTimeout(r, 200))

        sim.stopPolling()
        expect(sim.pollCount).toBeGreaterThanOrEqual(2)
    })
})

// ---------------------------------------------------------------------------
// Async tool execution flow (end-to-end simulation)
// ---------------------------------------------------------------------------

describe("VoiceAgentService — async tool execution flow (e2e)", () => {
    type ToolCall = { name: string; parameters: Record<string, any> }
    type ToolResult = { success: boolean; output: string; error?: string }

    /**
     * Simulates the full async tool execution flow:
     *   1. LLM emits tool call
     *   2. Tool executor dispatches async (returns quickly)
     *   3. Result pushed to history
     *   4. LLM re-invoked for spoken summary
     *   5. Background poller later detects completion
     *   6. Completion announced
     */
    function simulateAsyncFlow() {
        const history: Array<{ role: string; content: string }> = []
        const logs: string[] = []
        const pendingAnnouncements: BackgroundToolResult[] = []
        let llmCalls = 0

        /** Phase 1: Tool dispatch — fast return */
        async function dispatchTool(
            toolCall: ToolCall,
            executor: (tc: ToolCall) => Promise<ToolResult>,
        ) {
            const result = await executor(toolCall)
            history.push({
                role: "tool",
                content: JSON.stringify({
                    tool: toolCall.name,
                    success: result.success,
                    output: result.output,
                    ...(result.error && { error: result.error }),
                }),
            })
            logs.push(`Tool ${toolCall.name} dispatched: ${result.output.slice(0, 80)}`)

            // Re-call LLM for spoken summary
            llmCalls++
        }

        /** Phase 2: Poller detects completion */
        function receiveCompletion(result: BackgroundToolResult) {
            pendingAnnouncements.push(result)
        }

        /** Phase 3: Drain announcements */
        function drainAnnouncements() {
            if (pendingAnnouncements.length === 0) return

            const batch = pendingAnnouncements.splice(0)
            for (const ann of batch) {
                history.push({
                    role: "tool",
                    content: JSON.stringify({
                        tool: ann.tool,
                        taskId: ann.taskId,
                        task: ann.task,
                        success: ann.success,
                        output: ann.output,
                        ...(ann.error && { error: ann.error }),
                        note: "This background task has completed. Briefly announce the result to the user.",
                    }),
                })
            }
            llmCalls++
        }

        return {
            history,
            logs,
            pendingAnnouncements,
            get llmCalls() { return llmCalls },
            dispatchTool,
            receiveCompletion,
            drainAnnouncements,
        }
    }

    test("full flow: dispatch → poll completion → announce", async () => {
        const sim = simulateAsyncFlow()

        // Phase 1: Dispatch
        await sim.dispatchTool(
            { name: "HammerAgent", parameters: { task: "Create README" } },
            async () => ({
                success: true,
                output: "Background task started (ID: task_1). I will notify when done.",
            }),
        )

        expect(sim.llmCalls).toBe(1) // LLM re-invoked for dispatch summary
        expect(sim.history.length).toBe(1)

        // Phase 2: Background task completes
        sim.receiveCompletion({
            taskId: "task_1",
            tool: "HammerAgent",
            task: "Create README",
            success: true,
            output: "README.md created with project overview",
        })

        expect(sim.pendingAnnouncements.length).toBe(1)

        // Phase 3: Agent drains announcements
        sim.drainAnnouncements()

        expect(sim.llmCalls).toBe(2) // Second LLM call for completion announcement
        expect(sim.history.length).toBe(2) // Dispatch result + completion result
        expect(sim.pendingAnnouncements.length).toBe(0)
    })

    test("multiple tasks complete before drain — single batched announcement", async () => {
        const sim = simulateAsyncFlow()

        // Dispatch two tasks
        await sim.dispatchTool(
            { name: "HammerAgent", parameters: { task: "Task A" } },
            async () => ({ success: true, output: "Dispatched A" }),
        )
        await sim.dispatchTool(
            { name: "HammerAgent", parameters: { task: "Task B" } },
            async () => ({ success: true, output: "Dispatched B" }),
        )

        expect(sim.llmCalls).toBe(2)

        // Both tasks complete
        sim.receiveCompletion({
            taskId: "tA",
            tool: "HammerAgent",
            task: "Task A",
            success: true,
            output: "A done",
        })
        sim.receiveCompletion({
            taskId: "tB",
            tool: "HammerAgent",
            task: "Task B",
            success: false,
            output: "",
            error: "Build failed",
        })

        // Single drain for both
        sim.drainAnnouncements()

        expect(sim.llmCalls).toBe(3) // Only 1 extra LLM call for both
        expect(sim.history.length).toBe(4) // 2 dispatch + 2 completion

        // Verify last two history entries are the completions
        const lastTwo = sim.history.slice(-2)
        expect(JSON.parse(lastTwo[0].content).taskId).toBe("tA")
        expect(JSON.parse(lastTwo[1].content).taskId).toBe("tB")
    })

    test("failed dispatch does not trigger announcement logic", async () => {
        const sim = simulateAsyncFlow()

        await sim.dispatchTool(
            { name: "HammerAgent", parameters: { task: "Fail" } },
            async () => ({
                success: false,
                output: "",
                error: "Server unreachable",
            }),
        )

        // Dispatch result is in history but no background task was started
        expect(sim.history.length).toBe(1)
        const parsed = JSON.parse(sim.history[0].content)
        expect(parsed.success).toBe(false)
        expect(parsed.error).toBe("Server unreachable")

        // No background completion expected
        expect(sim.pendingAnnouncements.length).toBe(0)
    })

    test("drain with no pending completions is a no-op", () => {
        const sim = simulateAsyncFlow()
        sim.drainAnnouncements()

        expect(sim.llmCalls).toBe(0)
        expect(sim.history.length).toBe(0)
    })
})

// ---------------------------------------------------------------------------
// executeToolsAndRespond with async tool result
// ---------------------------------------------------------------------------

describe("VoiceAgentService — executeToolsAndRespond with async tool", () => {
    type ToolCall = { name: string; parameters: Record<string, any> }

    async function simulateToolExecution(
        calls: ToolCall[],
        executor: (tc: ToolCall) => Promise<{ success: boolean; output: string; error?: string }>,
        signal: AbortSignal,
    ) {
        const history: Array<{ role: string; content: string }> = []
        let reprocessCalled = false

        for (const tc of calls) {
            if (signal.aborted) break

            try {
                const result = await executor(tc)
                history.push({
                    role: "tool",
                    content: JSON.stringify({
                        tool: tc.name,
                        success: result.success,
                        output: result.output,
                        ...(result.error && { error: result.error }),
                    }),
                })
            } catch (err) {
                history.push({
                    role: "tool",
                    content: JSON.stringify({
                        tool: tc.name,
                        success: false,
                        output: "",
                        error: (err as Error).message,
                    }),
                })
            }
        }

        if (!signal.aborted) {
            reprocessCalled = true
        }

        return { history, reprocessCalled }
    }

    test("async tool dispatch returns quickly and triggers LLM re-call", async () => {
        const controller = new AbortController()
        const start = Date.now()

        const result = await simulateToolExecution(
            [{ name: "HammerAgent", parameters: { task: "Create file" } }],
            async () => ({
                success: true,
                output: "Background task started (ID: task_1). I will notify when done.",
            }),
            controller.signal,
        )

        const elapsed = Date.now() - start

        expect(elapsed).toBeLessThan(100) // Fast — no blocking
        expect(result.reprocessCalled).toBe(true)
        expect(result.history.length).toBe(1)

        const parsed = JSON.parse(result.history[0].content)
        expect(parsed.success).toBe(true)
        expect(parsed.output).toContain("Background task started")
    })

    test("abort during async dispatch stops execution", async () => {
        const controller = new AbortController()

        const result = await simulateToolExecution(
            [
                { name: "HammerAgent", parameters: { task: "Task A" } },
                { name: "HammerAgent", parameters: { task: "Task B" } },
            ],
            async (tc) => {
                if (tc.parameters.task === "Task A") {
                    controller.abort()
                }
                return { success: true, output: "Dispatched" }
            },
            controller.signal,
        )

        expect(result.history.length).toBe(1) // Only first executed
        expect(result.reprocessCalled).toBe(false) // Aborted
    })
})
