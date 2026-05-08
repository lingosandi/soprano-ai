/**
 * Precise token estimator using tiktoken (cl100k_base encoding).
 *
 * Separated into its own module so environments can opt into the tiktoken WASM
 * dependency explicitly. Browser runtimes should reach it through async
 * estimator helpers instead of importing it from hot paths.
 */

import { encoding_for_model, type Tiktoken } from "tiktoken"
import type { TokenEstimator } from "./memory-layer"

export class TiktokenEstimator implements TokenEstimator {
    private encoder: Tiktoken | null

    constructor() {
        try {
            this.encoder = encoding_for_model("gpt-4")
        } catch (err) {
            throw new Error(
                `Failed to initialize tiktoken encoder (WASM may not be available in this environment): ${err instanceof Error ? err.message : err}`,
            )
        }
    }

    estimateTokens(text: string): number {
        if (!this.encoder) throw new Error("TiktokenEstimator has been disposed")
        return this.encoder.encode(text).length
    }

    dispose(): void {
        if (this.encoder) {
            this.encoder.free()
            this.encoder = null
        }
    }
}
