/**
 * FunASRService — Tauri/web ASR using getUserMedia.
 *
 * Extends BaseASRService (shared DashScope bridge protocol) with
 * Web Audio API microphone capture: getUserMedia → AudioWorkletNode →
 * downsample to 16 kHz mono PCM16.
 */

import { BaseASRService } from "./base-asr-service"
import { ASR_SAMPLE_RATE } from "./config"
import { downsample, float32ToInt16 } from "./audio-helpers"

export interface FunASRServiceOptions {
    bridgeUrl?: string | (() => string)
    workletModuleUrl?: string
}

export class FunASRService extends BaseASRService {
    private readonly bridgeUrl?: string | (() => string)
    private readonly workletModuleUrl: string
    private mediaStream: MediaStream | null = null
    private audioContext: AudioContext | null = null
    private workletNode: AudioWorkletNode | null = null
    private sourceNode: MediaStreamAudioSourceNode | null = null
    private silentGainNode: GainNode | null = null
    private micMuted = false

    constructor(options: FunASRServiceOptions = {}) {
        super()
        this.bridgeUrl = options.bridgeUrl
        this.workletModuleUrl = options.workletModuleUrl ?? "/pcm-capture-processor.js"
    }

    protected override get logPrefix(): string {
        return "FunASR"
    }

    protected override getBridgeUrl(): string {
        const bridgeUrl = typeof this.bridgeUrl === "function"
            ? this.bridgeUrl()
            : this.bridgeUrl

        if (!bridgeUrl) {
            throw new Error("FunASRService requires a bridgeUrl option")
        }

        return bridgeUrl
    }

    protected override async startAudioCapture(): Promise<void> {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: { ideal: ASR_SAMPLE_RATE },
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            })
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e)
            throw new Error(`Microphone access denied: ${msg}`)
        }

        for (const track of this.mediaStream.getAudioTracks()) {
            track.enabled = !this.micMuted
        }

        this.audioContext = new AudioContext()
        const nativeSampleRate = this.audioContext.sampleRate

        // Register the AudioWorklet processor module (served from public/)
        await this.audioContext.audioWorklet.addModule(this.workletModuleUrl)

        this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream)
        this.workletNode = new AudioWorkletNode(this.audioContext, "pcm-capture-processor")

        // Receive Float32 PCM frames from the audio thread
        this.workletNode.port.onmessage = (ev: MessageEvent<Float32Array>) => {
            if (!this.isRunning) return
            if (this.isMicMuted) return
            const float32 = ev.data
            const downsampled = downsample(float32, nativeSampleRate, ASR_SAMPLE_RATE)
            const pcm16 = float32ToInt16(downsampled)
            this.pcmBuffer.push(pcm16)
        }

        this.sourceNode.connect(this.workletNode)
        // Connect through a silent gain node to keep the audio graph alive
        // without routing microphone audio to speakers (which would cause feedback)
        this.silentGainNode = this.audioContext.createGain()
        this.silentGainNode.gain.value = 0
        this.workletNode.connect(this.silentGainNode)
        this.silentGainNode.connect(this.audioContext.destination)
    }

    /** Mute/unmute the microphone by toggling the media stream track. */
    setMicMuted(muted: boolean): void {
        this.micMuted = muted
        if (this.mediaStream) {
            for (const track of this.mediaStream.getAudioTracks()) {
                track.enabled = !muted
            }
        }
        // Discard any buffered PCM captured before the mute to prevent
        // stale speech from reaching DashScope and triggering a transcript.
        if (muted) {
            this.pcmBuffer = []
        }
    }

    /** Whether the microphone is currently muted. */
    get isMicMuted(): boolean {
        return this.micMuted
    }

    protected override stopAudioCapture(): void {
        // Tell the worklet processor to stop returning true from process()
        this.workletNode?.port.postMessage("stop")
        this.workletNode?.disconnect()
        this.sourceNode?.disconnect()
        this.silentGainNode?.disconnect()
        this.workletNode = null
        this.sourceNode = null
        this.silentGainNode = null

        if (this.audioContext) {
            this.audioContext.close().catch(() => {})
            this.audioContext = null
        }

        if (this.mediaStream) {
            for (const track of this.mediaStream.getTracks()) {
                track.stop()
            }
            this.mediaStream = null
        }
    }
}
