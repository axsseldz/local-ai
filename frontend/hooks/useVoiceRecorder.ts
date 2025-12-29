"use client";

import { useCallback, useRef, useState } from "react";
import { floatTo16BitPCM, downsampleBuffer } from "@/lib/audio-utils";

type VoiceRecorderState = {
  isRecording: boolean;
  liveTranscript: string;
  micError: string;
};

type UseVoiceRecorderResult = VoiceRecorderState & {
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  toggleRecording: () => void;
  setLiveTranscript: (text: string) => void;
  setMicError: (text: string) => void;
};

type VoiceCallbacks = {
  onFinalText: (text: string) => void;
  onPartialText?: (text: string) => void;
};

export function useVoiceRecorder({ onFinalText, onPartialText }: VoiceCallbacks): UseVoiceRecorderResult {
  const [isRecording, setIsRecording] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [micError, setMicError] = useState("");

  const finalTranscriptRef = useRef<string>("");
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const stopRecording = useCallback(() => {
    if (!isRecording) return;
    setIsRecording(false);

    const textToSend = (finalTranscriptRef.current || "").trim();
    finalTranscriptRef.current = "";
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    processorRef.current = null;
    sourceRef.current = null;

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => { });
      audioCtxRef.current = null;
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;

    setLiveTranscript("");

    if (textToSend) {
      setTimeout(() => onFinalText(textToSend), 50);
    }
  }, [isRecording, onFinalText]);

  const startRecording = useCallback(async () => {
    setMicError("");
    setLiveTranscript("");

    if (isRecording) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setMicError("Your browser doesn't support microphone recording.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ws = new WebSocket("ws://localhost:8000/voice/stt/ws");
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "partial") {
            setLiveTranscript(msg.text || "");
            if (onPartialText) onPartialText(msg.text || "");
          } else if (msg.type === "final") {
            const finalText = (msg.text || "").trim();
            if (finalText) {
              finalTranscriptRef.current = finalText;
            }
          } else if (msg.type === "error") {
            setMicError(msg.message || "STT error");
          }
        } catch {
          // ignore
        }
      };

      ws.onerror = () => setMicError("WebSocket error (backend down?)");
      ws.onclose = () => { };

      const AudioCtx = (window.AudioContext || (window as any).webkitAudioContext);
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const wsNow = wsRef.current;
        if (!wsNow || wsNow.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0);
        const down = downsampleBuffer(input, audioCtx.sampleRate, 16000);
        const pcm16 = floatTo16BitPCM(down);

        wsNow.send(pcm16.buffer);
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);
    } catch (e: any) {
      setMicError("Microphone permission denied or not available.");
    }
  }, [isRecording, onPartialText]);

  const toggleRecording = useCallback(() => {
    if (isRecording) stopRecording();
    else startRecording();
  }, [isRecording, startRecording, stopRecording]);

  return {
    isRecording,
    liveTranscript,
    micError,
    startRecording,
    stopRecording,
    toggleRecording,
    setLiveTranscript,
    setMicError,
  };
}
