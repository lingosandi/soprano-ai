# soprano-ai

Browser-safe voice agent runtime for ASR, LLM streaming, Cartesia TTS playback, memory, and tool-calling orchestration.

## Secrets

This package does not ship API keys. Inject your own provider config, DashScope ASR key, and Cartesia key when creating `VoiceAgentService`.

```ts
import {
  AudioStreamPlayer,
  FunASRService,
  createSopranoVoiceAgent,
} from "soprano-ai"

const agent = createSopranoVoiceAgent({
  player: new AudioStreamPlayer(),
  asr: new FunASRService({ bridgeUrl: () => "ws://127.0.0.1:9231" }),
  apiKeys: {
    qwenApiKey: "YOUR_QWEN_DASHSCOPE_API_KEY",
    cartesiaApiKey: "YOUR_CARTESIA_API_KEY",
  },
})
```

## TTS voice and quality

Cartesia TTS defaults to the package voice and the original low-quality 8 kHz PCM stream. To choose a different Cartesia voice, pass its voice ID as `ttsVoiceId`. To opt into higher-quality 24 kHz PCM, pass the same quality to the default TTS config and to `AudioStreamPlayer`:

```ts
const ttsQuality = "high" as const
const ttsVoiceId = "f786b574-daa5-4673-aa0c-cbe3e8534c02"

const agent = createSopranoVoiceAgent({
  player: new AudioStreamPlayer({ quality: ttsQuality }),
  asr: new FunASRService({ bridgeUrl: () => "ws://127.0.0.1:9231" }),
  apiKeys: {
    qwenApiKey: "YOUR_QWEN_DASHSCOPE_API_KEY",
    cartesiaApiKey: "YOUR_CARTESIA_API_KEY",
  },
  ttsQuality,
  ttsVoiceId,
})
```

The chat orb or any other UI can subscribe with `agent.on(...)` and drive the service through `init()`, `startListening()`, `sendTextOnly()`, `interrupt()`, and `destroy()`.
