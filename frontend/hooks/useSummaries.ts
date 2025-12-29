"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { API_BASE, DEFAULT_TOP_K } from "@/constants/api";
import { buildConversationTranscript, buildSummaryPreview, buildSummaryPrompt, extractSummaryTitle, stripLeadingMeta, stripSummaryTitle, uid } from "@/lib/chat-utils";
import { SummaryItem, ChatMessage, Mode } from "@/types/chat";

type UseSummariesArgs = {
  messages: ChatMessage[];
  mode: Mode;
  model: string;
  conversationId: string;
};

export function useSummaries({ messages, mode, model, conversationId }: UseSummariesArgs) {
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [selectedSummaryId, setSelectedSummaryId] = useState<string | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem("jarvis_summaries");
      if (!raw) return;
      const parsed = JSON.parse(raw) as SummaryItem[];
      if (Array.isArray(parsed)) setSummaries(parsed);
    } catch {
      // ignore corrupted local data
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      window.localStorage.setItem("jarvis_summaries", JSON.stringify(summaries));
    } catch {
      // ignore storage failures
    }
  }, [summaries]);

  const sortedSummaries = useMemo(
    () => [...summaries].sort((a, b) => b.createdAt - a.createdAt),
    [summaries]
  );

  const selectedSummary = useMemo(
    () => summaries.find((s) => s.id === selectedSummaryId) || null,
    [summaries, selectedSummaryId]
  );

  const deleteSummary = useCallback((id: string) => {
    setSummaries((prev) => prev.filter((s) => s.id !== id));
    setSelectedSummaryId((prev) => (prev === id ? null : prev));
  }, []);

  const handleSelectSummary = useCallback((id: string | null) => {
    setSelectedSummaryId((prev) => (prev === id ? null : id));
    setIsSidebarOpen(true);
  }, []);

  const streamSummary = useCallback(
    async (prompt: string, summaryId: string) => {
      try {
        const res = await fetch(`${API_BASE}/ask/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({
            question: prompt,
            mode,
            top_k: DEFAULT_TOP_K,
            model,
            conversation_id: conversationId,
          }),
        });

        if (!res.ok || !res.body) throw new Error(`Streaming failed: ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

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

            if (eventName === "done") {
              setSummaries((prev) =>
                prev.map((s) => (s.id === summaryId ? { ...s, isLoading: false } : s))
              );
              continue;
            }

            let chunk = data;
            if (eventName === "meta") continue;
            if (eventName === "chunk" || eventName === "message") {
              try {
                const parsed = JSON.parse(data);
                if (typeof parsed === "string") chunk = parsed;
                else if (parsed && typeof parsed.content === "string") chunk = parsed.content;
                else chunk = "";
              } catch {
                if (chunk.startsWith(" ")) chunk = chunk.slice(1);
              }
            }

            setSummaries((prev) =>
              prev.map((s) => {
                if (s.id !== summaryId) return s;
                const nextContent = `${s.content}${chunk}`;
                const cleaned = stripLeadingMeta(nextContent);
                const nextTitle = s.title === "Generating title..." ? extractSummaryTitle(cleaned) : "";
                const displayContent = stripSummaryTitle(cleaned);
                return {
                  ...s,
                  content: cleaned,
                  title: nextTitle ? nextTitle : s.title,
                  preview: buildSummaryPreview(displayContent),
                };
              })
            );
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Summary failed";
        setSummaries((prev) =>
          prev.map((s) => (s.id === summaryId ? { ...s, isLoading: false, error: message } : s))
        );
      }
    },
    [conversationId, mode, model]
  );

  const createConversationSummary = useCallback(
    async (titleSeed: string) => {
      const transcript = buildConversationTranscript(messages);
      const summaryId = uid("s");
      const title = titleSeed || "Conversation Summary";
      const next: SummaryItem = {
        id: summaryId,
        title,
        preview: "Generating summary...",
        content: "",
        createdAt: Date.now(),
        isLoading: true,
      };

      setSummaries((prev) => [next, ...prev]);
      setIsSidebarOpen(true);

      const prompt = buildSummaryPrompt(transcript || "No prior conversation yet.", titleSeed);
      await streamSummary(prompt, summaryId);
    },
    [messages, streamSummary]
  );

  const closeSidebar = useCallback(() => {
    setIsSidebarOpen(false);
    setSelectedSummaryId(null);
  }, []);

  return {
    summaries,
    sortedSummaries,
    selectedSummary,
    selectedSummaryId,
    isSidebarOpen,
    setIsSidebarOpen,
    deleteSummary,
    handleSelectSummary,
    createConversationSummary,
    closeSidebar,
  };
}
