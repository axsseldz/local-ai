"use client";

import { useCallback, useRef, useState, MutableRefObject } from "react";
import { API_BASE, DEFAULT_TOP_K } from "@/constants/api";
import { normalizeSentenceForDedupe, speakText } from "@/lib/tts-utils";
import { ChatMessage, Mode } from "@/types/chat";

type UseChatStreamArgs = {
  mode: Mode;
  model: string;
  conversationIdRef: MutableRefObject<string>;
  ttsMuted: boolean;
  voice: SpeechSynthesisVoice | null;
  onSummaryCommand: (titleSeed: string) => Promise<void>;
};

export function useChatStream({
  mode,
  model,
  conversationIdRef,
  ttsMuted,
  voice,
  onSummaryCommand,
}: UseChatStreamArgs) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const spokenSentencesRef = useRef<Set<string>>(new Set());
  const ttsRemainderRef = useRef<string>("");

  const speak = useCallback(
    (text: string, opts?: { interrupt?: boolean }) => {
      speakText(text, { ttsMuted, voice, interrupt: opts?.interrupt });
    },
    [ttsMuted, voice]
  );

  const clearChat = useCallback(() => {
    setMessages([]);
    setError("");
    conversationIdRef.current = `c_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }, [conversationIdRef]);

  const askWithText = useCallback(
    async (rawText: string) => {
      const q = rawText.trim();
      if (!q || loading) return;

      const agentPattern = /(^|\s)\/summary(?=\s|$)/;
      if (agentPattern.test(q)) {
        const titleSeed = q.replace(agentPattern, " ").replace(/\s+/g, " ").trim();
        await onSummaryCommand(titleSeed);
        return;
      }

      setLoading(true);
      setError("");

      const userMsg: ChatMessage = {
        id: `u_${Date.now()}`,
        role: "user",
        content: q,
        createdAt: Date.now(),
        mode,
      };

      const assistantId = `a_${Date.now()}`;
      const pendingAssistant: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        isLoading: true,
      };

      spokenSentencesRef.current = new Set();
      ttsRemainderRef.current = "";
      setMessages((prev) => [...prev, userMsg, pendingAssistant]);

      try {
        const res = await fetch(`${API_BASE}/ask/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({ question: q, mode, top_k: DEFAULT_TOP_K, model, conversation_id: conversationIdRef.current }),
        });

        if (!res.ok || !res.body) throw new Error(`Streaming failed: ${res.status}`);

        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: "", isLoading: true } : m))
        );

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let metaApplied = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          const frames = buffer.split("\n\n");
          buffer = frames.pop() || "";

          for (const frame of frames) {
            const lines = frame.split("\n").filter(Boolean);
            let eventName = "message";
            const dataLines: string[] = [];

            for (const line of lines) {
              if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
              else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length));
            }

            const data = dataLines.join("\n");

            if (eventName === "meta") {
              try {
                const meta = JSON.parse(data);
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? { ...m, mode: meta.mode, task: meta.task, model: meta.model, sources: meta.sources || [] }
                      : m
                  )
                );
                metaApplied = true;
              } catch {
                // ignore meta parse errors
              }
              continue;
            }

            if (eventName === "done") {
              const tail = (ttsRemainderRef.current || "").trim();
              if (tail) {
                const key = normalizeSentenceForDedupe(tail);
                if (key && !spokenSentencesRef.current.has(key)) {
                  spokenSentencesRef.current.add(key);
                  speak(tail, { interrupt: false });
                }
              }
              ttsRemainderRef.current = "";
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantId ? { ...m, isLoading: false } : m))
              );
              continue;
            }

            let chunk = data;

            if (eventName === "chunk" || eventName === "message") {
              try {
                const parsed = JSON.parse(data);
                if (typeof parsed === "string") chunk = parsed;
              } catch {
                if (chunk.startsWith(" ")) chunk = chunk.slice(1);
              }
            }

            if (!metaApplied) metaApplied = true;

            ttsRemainderRef.current += chunk;

            while (true) {
              const r = ttsRemainderRef.current;
              const idx = r.search(/[.!?]/);

              if (idx === -1) break;

              let end = idx + 1;
              while (end < r.length && /[.!?]/.test(r[end])) end++;

              if (end < r.length && !/\s/.test(r[end])) break;

              const sentence = r.slice(0, end).trim();
              ttsRemainderRef.current = r.slice(end).trimStart();

              if (sentence.length >= 3) {
                const key = normalizeSentenceForDedupe(sentence);
                if (key && !spokenSentencesRef.current.has(key)) {
                  spokenSentencesRef.current.add(key);
                  speak(sentence, { interrupt: false });
                }
              }
            }

            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: (m.content || "") + chunk, isLoading: false }
                  : m
              )
            );
          }
        }
      } catch {
        const msg = "Could not reach backend. Is FastAPI running on http://localhost:8000 ?";
        setError(msg);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: msg, error: true, isLoading: false } : m
          )
        );
      } finally {
        setLoading(false);
      }
    },
    [conversationIdRef, loading, mode, model, onSummaryCommand, speak]
  );

  return {
    messages,
    setMessages,
    loading,
    error,
    setError,
    askWithText,
    clearChat,
  };
}
