import type { TokenEstimator } from "./memory-layer"
import { TiktokenEstimator } from "./tiktoken-estimator"

/**
 * Default voice-memory estimator for Bun/web environments.
 */
export async function createVoiceTokenEstimator(): Promise<TokenEstimator> {
    return new TiktokenEstimator()
}