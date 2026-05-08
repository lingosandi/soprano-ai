import type { TokenEstimator } from "./memory-layer"
import { TiktokenEstimator } from "./tiktoken-estimator"

/**
 * Default voice-memory estimator for Bun/web environments.
 *
 * Kept in a separate module so React Native can resolve the
 * platform-specific .native variant that never imports tiktoken.
 */
export async function createVoiceTokenEstimator(): Promise<TokenEstimator> {
    return new TiktokenEstimator()
}