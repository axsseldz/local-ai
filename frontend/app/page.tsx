"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

import { DocsMenu } from "@/components/DocsMenu";
import { IndexStatusCard } from "@/components/IndexStatusCard";
import { MessageList } from "@/components/MessageList";
import { MetricsCard } from "@/components/MetricsCard";
import { ModelMenu } from "@/components/ModelMenu";
import { PromptInput } from "@/components/PromptInput";
import RadarBackground from "@/components/RadarBackground";
import { SummaryDrawer } from "@/components/SummaryDrawer";
import { modeIcons, modeLabels } from "@/constants/modes";
import { useVoiceRecorder } from "@/hooks/useVoiceRecorder";
import {
  buildConversationTranscript,
  buildSummaryPreview,
  buildSummaryPrompt,
  enforceSingleAgentCommand,
  extractSummaryTitle,
  formatAssistantContent,
  formatBytes,
  renderPromptPreview,
  stripLeadingMeta,
  stripSummaryTitle,
  uid,
} from "@/lib/chat-utils";
import type { ChatMessage, DocItem, IndexStatus, MetricsResponse, Mode, SummaryItem } from "@/types/chat";

const FALLBACK_MODELS = ["llama3.1:8b", "qwen2.5-coder:7b", "gpt-oss:20b"];
const SUMMARY_COMMAND = /(^|\s)\/summary(?=\s|$)/;
const DOCS_BUTTON_LABEL = "Documents";

function stripForSpeech(input: string) {
  let t = input || "";

  t = t.replace(/```[\s\S]*?```/g, " ");
  t = t.replace(/`([^`]+)`/g, "$1");
  t = t.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  t = t.replace(/\[S\d+\]/g, "");
  t = t.replace(/\s+/g, " ").trim();

  if (/^loading/i.test(t)) return "";

  return t;
}

function normalizeSentenceForDedupe(s: string) {
  return stripForSpeech(s)
    .toLowerCase()
    .replace(/\s+/g, " ")
    .replace(/\s+([.!?,])/g, "$1")
    .trim();
}

function formatSummaryStamp(ts: number) {
  const date = new Date(ts);
  const day = date.toLocaleDateString([], { month: "short", day: "2-digit" });
  const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return `${day} · ${time}`;
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("general");
  const [model, setModel] = useState(FALLBACK_MODELS[0]);
  const [modelOptions, setModelOptions] = useState<string[]>(FALLBACK_MODELS);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexStarting, setIndexStarting] = useState(false);
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [summaries, setSummaries] = useState<SummaryItem[]>([]);
  const [selectedSummaryId, setSelectedSummaryId] = useState<string | null>(null);
  const [ttsMuted, setTtsMuted] = useState(true);

  const spokenRef = useRef<string>("");
  const chatRef = useRef<HTMLDivElement | null>(null);
  const questionInputRef = useRef<HTMLTextAreaElement | null>(null);
  const spokenSentencesRef = useRef<Set<string>>(new Set());
  const ttsRemainderRef = useRef<string>("");
  const jarvisVoiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const conversationIdRef = useRef(uid("c"));
  const { isRecording, micError, toggleRecording } = useVoiceRecorder({
    onFinalText: (text) => {
      setQuestion(text);
      setTimeout(() => askWithText(text), 50);
    },
    onPartialText: (text) => setQuestion(text),
  });


  useEffect(() => {
    const id = requestAnimationFrame(() => {
      chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  useEffect(() => {
    let alive = true;

    const loadModels = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/models");
        if (!res.ok) throw new Error(`models request failed: ${res.status}`);
        const data = await res.json();
        const list = Array.isArray(data?.models)
          ? data.models.filter((m: unknown): m is string => typeof m === "string")
          : [];
        if (!alive) return;
        if (list.length) {
          setModelOptions(list);
          setModel((prev) => {
            if (prev && list.includes(prev)) return prev;
            const defaultModel =
              typeof data?.default === "string" && list.includes(data.default) ? data.default : null;
            return defaultModel || list[0] || prev || FALLBACK_MODELS[0];
          });
        }
      } catch {
        if (!alive) return;
        setModelOptions((prev) => (prev.length ? prev : FALLBACK_MODELS));
      }
    };

    loadModels();
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!("speechSynthesis" in window)) return;

    const pickJarvis = () => {
      const voices = window.speechSynthesis.getVoices() || [];
      const exact = voices.find(v => v.name === "Google UK English Male" && v.lang === "en-GB");
      const fallback =
        exact ||
        voices.find(v => v.name.includes("Google UK English Male")) ||
        voices.find(v => v.lang === "en-GB") ||
        voices.find(v => v.lang?.startsWith("en")) ||
        voices[0] ||
        null;

      jarvisVoiceRef.current = fallback;
    };

    pickJarvis();
    window.speechSynthesis.onvoiceschanged = pickJarvis;

    return () => {
      window.speechSynthesis.onvoiceschanged = null;
    };
  }, []);

  useEffect(() => {
    fetchIndexStatus();

    const t = setInterval(() => {
      const running = indexStatus?.is_indexing;
      if (running) fetchIndexStatus();
    }, 1200);

    const slow = setInterval(() => {
      const running = indexStatus?.is_indexing;
      if (!running) fetchIndexStatus();
    }, 8000);

    return () => {
      clearInterval(t);
      clearInterval(slow);
    };
  }, [indexStatus?.is_indexing]);

  const canAsk = useMemo(() => question.trim().length > 0 && !loading, [question, loading]);

  async function ask() {
    await askWithText(question);
  }

  function speakText(rawText: string, opts?: { interrupt?: boolean }) {
    if (typeof window === "undefined") return;
    if (ttsMuted) return;
    if (!("speechSynthesis" in window)) return;

    const text = stripForSpeech(rawText);
    if (!text) return;


    if (spokenRef.current === text) return;
    spokenRef.current = text;

    const interrupt = opts?.interrupt ?? false;

    try {
      if (interrupt) window.speechSynthesis.cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.98;
      utterance.pitch = 0.85;
      utterance.volume = 1.0;

      const v = jarvisVoiceRef.current;
      if (v) utterance.voice = v;

      window.speechSynthesis.speak(utterance);
    } catch {
      // ignore
    }
  }

  function toggleMute() {
    setTtsMuted((m) => {
      const next = !m;
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      spokenRef.current = "";
      return next;
    });
  }


  async function fetchDocs() {
    try {
      const r = await fetch("http://localhost:8000/docs/list");
      const data = await r.json();
      setDocs(data.docs || []);
      if (!selectedDoc && data.docs?.length) {
        setSelectedDoc(data.docs[0].name);
      }
    } catch {
      // ignore
    }
  }

  async function uploadDoc(file: File) {
    try {
      setUploading(true);
      const fd = new FormData();
      fd.append("file", file);

      const r = await fetch("http://localhost:8000/docs/upload", {
        method: "POST",
        body: fd,
      });

      if (!r.ok) throw new Error("upload failed");

      await fetchDocs();
      setTimeout(fetchIndexStatus, 400);
    } finally {
      setUploading(false);
    }
  }

  async function fetchIndexStatus() {
    try {
      const r = await fetch("http://localhost:8000/index/status");
      const data = (await r.json()) as IndexStatus;
      setIndexStatus(data);
    } catch {
      // ignore
    }
  }

  async function startIndexNow() {
    try {
      setIndexStarting(true);
      await fetch("http://localhost:8000/index/run", { method: "POST" });
      setTimeout(fetchIndexStatus, 400);
    } finally {
      setIndexStarting(false);
    }
  }

  async function askWithText(text: string) {
    const q = text.trim();
    if (!q || loading) return;

    if (SUMMARY_COMMAND.test(q)) {
      const titleSeed = q.replace(SUMMARY_COMMAND, " ").replace(/\s+/g, " ").trim();
      setQuestion("");
      await createConversationSummary(titleSeed);
      return;
    }

    setLoading(true);
    setError("");

    const userMsg: ChatMessage = {
      id: uid("u"),
      role: "user",
      content: q,
      createdAt: Date.now(),
      mode,
    };

    const assistantId = uid("a");
    const pendingAssistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      isLoading: true,
    };

    spokenRef.current = "";
    spokenSentencesRef.current = new Set();
    ttsRemainderRef.current = "";
    setMessages((prev) => [...prev, userMsg, pendingAssistant]);
    setQuestion("");

    try {
      const res = await fetch("http://localhost:8000/ask/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ question: q, mode, top_k: 10, model, conversation_id: conversationIdRef.current }),
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
            } catch { }
            continue;
          }

          if (eventName === "done") {
            const tail = (ttsRemainderRef.current || "").trim();
            if (tail) {
              const key = normalizeSentenceForDedupe(tail);
              if (key && !spokenSentencesRef.current.has(key)) {
                spokenSentencesRef.current.add(key);
                speakText(tail, { interrupt: false });
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
                speakText(sentence, { interrupt: false });
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
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  }

  function onQuestionChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    const raw = e.currentTarget.value;
    const cursor = e.currentTarget.selectionStart ?? raw.length;
    const sanitized = enforceSingleAgentCommand(raw);
    const shouldPad = cursor === sanitized.length && /(^|\s)\/summary$/.test(sanitized);
    if (shouldPad) {
      const padded = `${sanitized} `;
      setQuestion(padded);
      requestAnimationFrame(() => {
        if (questionInputRef.current) {
          questionInputRef.current.selectionStart = padded.length;
          questionInputRef.current.selectionEnd = padded.length;
        }
      });
      return;
    }
    setQuestion(sanitized);
    if (sanitized !== raw) {
      requestAnimationFrame(() => {
        if (questionInputRef.current) {
          const nextPos = Math.min(cursor - 6, sanitized.length);
          questionInputRef.current.selectionStart = nextPos;
          questionInputRef.current.selectionEnd = nextPos;
        }
      });
    }
  }

  function deleteSummary(id: string) {
    setSummaries((prev) => prev.filter((s) => s.id !== id));
    setSelectedSummaryId((prev) => (prev === id ? null : prev));
  }

  function handleSelectSummary(id: string | null) {
    setSelectedSummaryId((prev) => (prev === id ? null : id));
    setIsSidebarOpen(true);
  }

  function closeSidebar() {
    setIsSidebarOpen(false);
    setSelectedSummaryId(null);
  }

  async function streamSummary(prompt: string, summaryId: string) {
    try {
      const res = await fetch("http://localhost:8000/ask/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          question: prompt,
          mode,
          top_k: 10,
          model,
          conversation_id: conversationIdRef.current,
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
  }

  async function createConversationSummary(titleSeed: string) {
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

    const prompt = buildSummaryPrompt(
      transcript || "No prior conversation yet.",
      titleSeed
    );
    await streamSummary(prompt, summaryId);
  }

  function clearChat() {
    setMessages([]);
    setError("");
    conversationIdRef.current = uid("c");
  }

  useEffect(() => {
    fetchDocs();
  }, []);

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

  useEffect(() => {
    if (indexStatus?.state === "ok" && !indexStatus?.is_indexing) {
      fetchDocs();
    }
  }, [indexStatus?.last_finished_at]);

  useEffect(() => {
    let alive = true;
    const fetchMetrics = async () => {
      try {
        const res = await fetch("http://localhost:8000/api/metrics");
        if (!res.ok) return;
        const data = (await res.json()) as MetricsResponse;
        if (alive) setMetrics(data);
      } catch {
        if (alive) setMetrics(null);
      }
    };

    fetchMetrics();
    const id = setInterval(fetchMetrics, 1500);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const indexStats = indexStatus?.stats;
  const indexProgress =
    indexStats && indexStats.files_on_disk > 0
      ? Math.min(
          100,
          Math.round(
            ((indexStats.indexed_files + indexStats.skipped_files + indexStats.deleted_files) /
              indexStats.files_on_disk) *
              100
          )
        )
      : null;
  const sortedSummaries = useMemo(
    () => [...summaries].sort((a, b) => b.createdAt - a.createdAt),
    [summaries]
  );
  const selectedSummary = useMemo(
    () => sortedSummaries.find((s) => s.id === selectedSummaryId) || null,
    [sortedSummaries, selectedSummaryId]
  );

  return (
    <div
      className="app-shell relative min-h-screen text-slate-100 overflow-y-scroll flex flex-col"
      style={{ scrollbarGutter: "stable" }}
    >
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-120 w-120 rounded-full blur-3xl opacity-35 bg-linear-to-r from-[#0f2a24] via-[#0c1f1c] to-transparent" />
        <div className="absolute -bottom-40 -right-40 h-160 w-160 rounded-full blur-3xl opacity-35 bg-linear-to-r from-[#0d2420] via-[#0a1a18] to-transparent" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.02),transparent_60%)]" />
        <div className="absolute inset-0 opacity-[0.1] bg-[radial-gradient(circle_at_20%_15%,rgba(45,212,191,0.2),transparent_30%),radial-gradient(circle_at_85%_10%,rgba(32,194,170,0.2),transparent_28%),radial-gradient(circle_at_40%_78%,rgba(20,130,120,0.2),transparent_30%)]" />
        <div className="absolute inset-0 opacity-[0.06] scan-grid grid-warp" />
        <div className="absolute inset-0 scanlines" />
        <div className="absolute inset-0 noise-shimmer" />
        <div className="absolute inset-0 vignette-overlay" />
        <RadarBackground />
      </div>

      <SummaryDrawer
        isOpen={isSidebarOpen}
        summaries={sortedSummaries}
        selectedSummary={selectedSummary}
        onSelectSummary={handleSelectSummary}
        onDeleteSummary={deleteSummary}
        onClose={closeSidebar}
        formatSummaryStamp={formatSummaryStamp}
        stripSummaryTitle={stripSummaryTitle}
      />

      <div className="absolute top-4 left-4 z-20">
        <div className="mt-6 flex items-center gap-3">
          <button
            type="button"
            onClick={() => setIsSidebarOpen(true)}
            className="inline-flex h-11 w-11 mb-1 items-center justify-center rounded-2xl border-0"
            aria-label="Open sidebar"
          >
            <svg
              viewBox="0 0 24 24"
              className="h-7 w-7"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M4 6h16" />
              <path d="M4 12h16" />
              <path d="M4 18h16" />
            </svg>
          </button>
          <ModelMenu model={model} modelOptions={modelOptions} onChange={setModel} />
        </div>
      </div>

      <div className="absolute top-4 right-4 z-10 isolate">
        <div className="inline-flex flex-col items-stretch gap-2 bg-transparent">
          <div className="mt-6 inline-flex items-center gap-2 rounded-2xl px-2.5 py-1.5 text-[11px] text-slate-300 glass-panel glass-panel--thin neon-edge">
            <DocsMenu docs={docs} label={DOCS_BUTTON_LABEL} />
            <label
              className="rounded-lg border border-slate-800/70 bg-transparent px-2 py-1 text-[11px] font-semibold text-slate-200 transition hover:border-emerald-300/60 hover:text-emerald-100"
              title="Upload a document into data/documents and auto-index"
            >
              <span className="inline-flex items-center gap-1.5">
                <svg
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 3v12" />
                  <path d="M7 8l5-5 5 5" />
                  <path d="M5 21h14" />
                </svg>
                <span>Upload</span>
              </span>
              <input
                type="file"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) uploadDoc(f);
                  e.currentTarget.value = "";
                }}
                disabled={uploading}
              />
            </label>
          </div>

          <IndexStatusCard
            indexStatus={indexStatus}
            indexProgress={indexProgress}
            indexStarting={indexStarting}
            onStartIndex={startIndexNow}
          />

          <MetricsCard
            metrics={metrics}
            formatBytes={formatBytes}
            modelFallback={model}
          />
        </div>
      </div>

      <div className="relative mx-auto w-[90vw] max-w-250 px-4 pt-2 pb-4 flex-1 flex flex-col gap-3 min-h-0">
        {/* error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-2xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-300 backdrop-blur glass-panel"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* chat history above prompt */}
        <div
          className="mx-auto w-full max-w-250 p-4 mt-4 flex-1 min-h-80 min-w-0 overflow-y-auto scroll-smooth hide-scrollbar"
          ref={chatRef}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const f = e.dataTransfer.files?.[0];
            if (f) uploadDoc(f);
          }}
        >
          <MessageList messages={messages} formatAssistantContent={formatAssistantContent} />
        </div>

        <PromptInput
          question={question}
          mode={mode}
          modeLabels={modeLabels}
          modeIcons={modeIcons}
          renderPromptPreview={renderPromptPreview}
          onQuestionChange={onQuestionChange}
          onKeyDown={onKeyDown}
          toggleRecording={toggleRecording}
          isRecording={isRecording}
          ttsMuted={ttsMuted}
          onToggleMute={toggleMute}
          loading={loading}
          canAsk={canAsk}
          ask={ask}
          clearChat={clearChat}
          micError={micError}
          setMode={setMode}
          questionInputRef={questionInputRef as React.RefObject<HTMLTextAreaElement>}
        />
      </div>
    </div>
  );
}
