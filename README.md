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

The chat orb or any other UI can subscribe with `agent.on(...)` and drive the service through `init()`, `startListening()`, `sendTextOnly()`, `interrupt()`, and `destroy()`.
