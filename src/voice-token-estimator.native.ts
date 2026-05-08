import { CharTokenEstimator, type TokenEstimator } from "./memory-layer"

/**
 * React Native cannot load tiktoken's WASM payload, so use the
 * lightweight character-based estimator on native/mobile builds.
 */
export async function createVoiceTokenEstimator(): Promise<TokenEstimator> {
    return new CharTokenEstimator()
}