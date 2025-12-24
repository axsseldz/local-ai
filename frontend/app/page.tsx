"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import RadarBackground from "@/components/RadarBackground";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type Mode = "local" | "general" | "search";

type Source = {
  label: string;
  doc_path?: string;
  chunk_index?: number;
  score?: number | null;
  title?: string;
  url?: string;
  snippet?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: number;
  mode?: string;
  task?: string;
  model?: string;
  sources?: Source[];
  error?: boolean;
  isLoading?: boolean;
};

type IndexStatus = {
  state: "idle" | "running" | "ok" | "error";
  is_indexing: boolean;
  last_trigger?: string | null;
  last_started_at?: string | null;
  last_finished_at?: string | null;
  last_error?: string | null;
  stats?: {
    files_on_disk: number;
    indexed_files: number;
    skipped_files: number;
    deleted_files: number;
    chunks_indexed: number;
  } | null;
};

type DocItem = {
  name: string;
  path: string;
  size: number;
  modified_at: string;
};

type MetricsResponse = {
  system?: {
    memory_used_bytes?: number | null;
    memory_total_bytes?: number | null;
    swap_used_bytes?: number | null;
    swap_total_bytes?: number | null;
    cpu_percent?: number | null;
    gpu_util_percent?: number | null;
    metal_supported?: boolean | null;
  };
  llm?: {
    tokens_per_second?: number | null;
    ttft_ms?: number | null;
    context_chars?: number | null;
    last_updated?: number | null;
  };
  model?: {
    name?: string | null;
    quantization?: string | null;
    backend?: string | null;
  };
};

function formatAssistantContent(message: ChatMessage) {

  if (message.role !== "assistant") return message.content;

  const text = message.content || "";
  if (!text) return "";

  return text.replace(/\r\n/g, "\n").replace(/\\n/g, "\n");
}

function cx(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function uid(prefix = "m") {
  return `${prefix}_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function floatTo16BitPCM(float32: Float32Array) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function downsampleBuffer(buffer: Float32Array, sampleRate: number, outRate = 16000) {
  if (outRate === sampleRate) return buffer;
  const ratio = sampleRate / outRate;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let offset = 0;
  for (let i = 0; i < newLen; i++) {
    const nextOffset = Math.round((i + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < buffer.length; j++) {
      sum += buffer[j];
      count++;
    }
    result[i] = count ? sum / count : 0;
    offset = nextOffset;
  }
  return result;
}

export default function Page() {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("general");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexStarting, setIndexStarting] = useState(false);
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>("");
  const [uploading, setUploading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState("");
  const [showModeMenu, setShowModeMenu] = useState(false);
  const [showDocsMenu, setShowDocsMenu] = useState(false);
  const [docsMenuPos, setDocsMenuPos] = useState<{ top: number; left: number } | null>(null);
  const [modeMenuPos, setModeMenuPos] = useState<{ top: number; left: number } | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);

  const spokenRef = useRef<string>("");
  const finalTranscriptRef = useRef<string>("");
  const chatRef = useRef<HTMLDivElement | null>(null);
  const modeMenuRef = useRef<HTMLDivElement | null>(null);
  const docsMenuRef = useRef<HTMLDivElement | null>(null);
  const docsButtonRef = useRef<HTMLButtonElement | null>(null);
  const modeButtonRef = useRef<HTMLButtonElement | null>(null);
  const modeMenuPortalRef = useRef<HTMLDivElement | null>(null);
  const docsMenuPortalRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const spokenSentencesRef = useRef<Set<string>>(new Set());
  const ttsRemainderRef = useRef<string>("");
  const [ttsMuted, setTtsMuted] = useState(true);
  const [liveTranscript, setLiveTranscript] = useState("");


  useEffect(() => {
    const id = requestAnimationFrame(() => {
      chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  useEffect(() => {
    if (!showDocsMenu) {
      setDocsMenuPos(null);
      return;
    }

    const updatePos = () => {
      const btn = docsButtonRef.current;
      if (!btn || typeof window === "undefined") return;
      const rect = btn.getBoundingClientRect();
      const menuWidth = 224;
      const left = Math.max(8, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - 8));
      const top = rect.bottom + 8;
      setDocsMenuPos({ top, left });
    };

    updatePos();
    window.addEventListener("resize", updatePos);
    window.addEventListener("scroll", updatePos, true);
    return () => {
      window.removeEventListener("resize", updatePos);
      window.removeEventListener("scroll", updatePos, true);
    };
  }, [showDocsMenu]);

  useEffect(() => {
    if (!showModeMenu) {
      setModeMenuPos(null);
      return;
    }

    const updatePos = () => {
      const btn = modeButtonRef.current;
      if (!btn || typeof window === "undefined") return;
      const rect = btn.getBoundingClientRect();
      const menuWidth = 240;
      const menuHeight = 176;
      const left = Math.max(8, Math.min(rect.left, window.innerWidth - menuWidth - 8));
      const top = Math.max(8, rect.top - menuHeight - 8);
      setModeMenuPos({ top, left });
    };

    updatePos();
    window.addEventListener("resize", updatePos);
    window.addEventListener("scroll", updatePos, true);
    return () => {
      window.removeEventListener("resize", updatePos);
      window.removeEventListener("scroll", updatePos, true);
    };
  }, [showModeMenu]);

  useEffect(() => {
    if (!showModeMenu && !showDocsMenu) return;
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node;
      const hitMode = modeMenuRef.current?.contains(target) || modeMenuPortalRef.current?.contains(target);
      const hitDocs = docsMenuRef.current?.contains(target) || docsMenuPortalRef.current?.contains(target);
      if (!hitMode) setShowModeMenu(false);
      if (!hitDocs) setShowDocsMenu(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [showModeMenu, showDocsMenu]);

  const jarvisVoiceRef = useRef<SpeechSynthesisVoice | null>(null);

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

  function formatBytes(bytes?: number | null) {
    if (bytes == null || Number.isNaN(bytes)) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let v = bytes;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
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

  async function startRecording() {
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
            setQuestion(msg.text || "");
          } else if (msg.type === "final") {
            const finalText = (msg.text || "").trim();
            if (finalText) {
              finalTranscriptRef.current = finalText;
              setQuestion(finalText);
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
  }

  function stopRecording() {
    if (!isRecording) return;
    setIsRecording(false);

    const textToSend = (finalTranscriptRef.current || question).trim();
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
      setTimeout(() => askWithText(textToSend), 50);
    }
  }

  function toggleRecording() {
    if (isRecording) stopRecording();
    else startRecording();
  }

  async function askWithText(text: string) {
    const q = text.trim();
    if (!q || loading) return;

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
        body: JSON.stringify({ question: q, mode, top_k: 10 }),
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

  function clearChat() {
    setMessages([]);
    setError("");
  }

  useEffect(() => {
    fetchDocs();
  }, []);

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

  const modeLabels: Record<Mode, string> = {
    local: "Local",
    general: "General",
    search: "Search",
  };
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
  const docsButtonLabel = "Documents";
  const memUsed = metrics?.system?.memory_used_bytes ?? null;
  const memTotal = metrics?.system?.memory_total_bytes ?? null;
  const memRatio = memUsed && memTotal ? memUsed / memTotal : null;
  const memPressure =
    memRatio == null ? "unknown" : memRatio < 0.75 ? "green" : memRatio < 0.9 ? "yellow" : "red";
  const swapUsed = metrics?.system?.swap_used_bytes ?? null;
  const cpuPercent = metrics?.system?.cpu_percent ?? null;
  const gpuUtil = metrics?.system?.gpu_util_percent ?? null;
  const metalSupported = metrics?.system?.metal_supported ?? null;
  const tps = metrics?.llm?.tokens_per_second ?? null;
  const ttft = metrics?.llm?.ttft_ms ?? null;
  const contextChars = metrics?.llm?.context_chars ?? null;
  const modelName = metrics?.model?.name ?? null;
  const modelQuant = metrics?.model?.quantization ?? null;
  const modelBackend = metrics?.model?.backend ?? null;

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

      <div className="absolute top-4 left-4 z-20" />

      <div className="absolute top-4 right-4 z-30 isolate">
        <div className="inline-flex flex-col items-stretch gap-2 bg-transparent">
          <div className="mt-6 inline-flex items-center gap-2 rounded-2xl px-2.5 py-1.5 text-[11px] text-slate-300 glass-panel glass-panel--thin neon-edge">
            {/* Docs dropdown */}
            <div className="relative z-50" ref={docsMenuRef}>
              <button
                type="button"
                onClick={() => setShowDocsMenu((v) => !v)}
                ref={docsButtonRef}
                className={cx(
                  "inline-flex items-center gap-2 rounded-lg border bg-transparent px-2 py-1 text-[11px] font-semibold text-slate-200 transition",
                  showDocsMenu
                    ? "border-emerald-400/60 bg-emerald-500/15 text-emerald-100"
                    : "border-slate-800/70 hover:border-emerald-300/60 hover:text-emerald-100"
                )}
                title="Documents in data/documents"
              >
                <span className="max-w-30 truncate">{docsButtonLabel}</span>
                <svg
                  className={cx("h-3 w-3 text-emerald-200 transition-transform", showDocsMenu && "rotate-180")}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M6 9l6 6 6-6" />
                </svg>
              </button>
            </div>
            {showDocsMenu && docsMenuPos
              ? createPortal(
                <div
                  className="fixed z-9999 w-56 overflow-hidden rounded-xl glass-panel glass-panel--menu"
                  style={{ top: docsMenuPos.top, left: docsMenuPos.left }}
                  ref={docsMenuPortalRef}
                >
                  <div className="px-3 py-2 text-[11px] uppercase tracking-[0.35em] text-slate-400">
                    Documents
                  </div>
                  <div className="max-h-56 overflow-y-auto hide-scrollbar">
                    {docs.length === 0 ? (
                      <div className="w-full px-3 py-2 text-left text-xs text-slate-400">
                        No docs
                      </div>
                    ) : (
                      docs.map((d) => (
                        <div
                          key={d.name}
                          className="flex w-full items-center justify-between px-3 py-2 text-xs font-semibold text-slate-200"
                        >
                          <span className="truncate">{d.name}</span>
                        </div>
                      ))
                    )}
                  </div>
                </div>,
                document.body
              )
              : null}

            {/* Upload (file picker) */}
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

            {/* Index now */}
            <button
              onClick={startIndexNow}
              disabled={!!indexStatus?.is_indexing || indexStarting}
              className={cx(
                "rounded-lg border px-2 py-1 text-[11px] font-semibold transition",
                indexStatus?.is_indexing || indexStarting
                  ? "border-slate-800/70 bg-transparent text-slate-500 cursor-not-allowed"
                  : "border-slate-800/70 bg-transparent text-slate-200 hover:border-emerald-300/60 hover:text-emerald-100"
              )}
              title="Run indexing now"
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
                  <path d="M4 4v6h6" />
                  <path d="M20 20v-6h-6" />
                  <path d="M20 8a8 8 0 0 0-14-4" />
                  <path d="M4 16a8 8 0 0 0 14 4" />
                </svg>
                <span>Index now</span>
              </span>
            </button>
          </div>

          <div className="relative z-0 w-full rounded-2xl px-3 py-2 glass-panel neon-edge">
            <div className="flex items-center justify-between gap-3 text-[11px] font-semibold text-slate-200">
              <span>Index status</span>
              <span
                className={cx(
                  "rounded-full px-2 py-0.5 text-[11px]",
                  indexStatus?.state === "running" && "bg-emerald-500/15 text-emerald-200",
                  indexStatus?.state === "ok" && "bg-teal-500/15 text-teal-200",
                  indexStatus?.state === "error" && "bg-rose-500/15 text-rose-200",
                  (!indexStatus || indexStatus?.state === "idle") && "bg-slate-700/30 text-slate-300"
                )}
              >
                {indexStatus?.is_indexing ? "Indexing…" : `${indexStatus?.state ?? "idle"}`}
              </span>
            </div>

            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-900/60">
              {indexStatus?.is_indexing ? (
                indexProgress !== null ? (
                  <div
                    className="h-full bg-linear-to-r from-emerald-500 via-teal-400 to-cyan-400 transition-[width] duration-300 index-bar"
                    style={{ width: `${indexProgress}%` }}
                  />
                ) : (
                  <div className="h-full w-2/5 bg-linear-to-r from-emerald-500 via-teal-400 to-cyan-400 index-bar-indeterminate" />
                )
              ) : (
                <div className="h-full w-full bg-linear-to-r from-emerald-500/40 to-teal-400/40" />
              )}
            </div>

            <div className="mt-2 text-[10px] text-slate-400">
              {indexStatus?.state === "error" && indexStatus?.last_error
                ? indexStatus.last_error
                : indexStats
                  ? `files ${indexStats.files_on_disk} · indexed ${indexStats.indexed_files} · skipped ${indexStats.skipped_files} · deleted ${indexStats.deleted_files}`
                  : "No index stats available yet."}
            </div>
          </div>

          <div className="relative z-0 w-full rounded-2xl px-3 py-2 glass-panel neon-edge">
            <div className="flex items-center justify-between gap-3 text-[11px] font-semibold text-slate-200">
              <span>LLM Resource Monitor</span>
              <span className="rounded-full bg-slate-700/30 px-2 py-0.5 text-[10px] text-slate-200">
                Live
              </span>
            </div>

            <div className="mt-2 space-y-2 text-[11px] text-slate-300">
              <div className="flex items-center justify-between">
                <span>Memory</span>
                <span className="text-slate-100">
                  {formatBytes(memUsed)} / {formatBytes(memTotal)}
                </span>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900/60">
                <div
                  className={cx(
                    "h-full transition-[width]",
                    memPressure === "green" && "bg-emerald-400/70",
                    memPressure === "yellow" && "bg-amber-400/70",
                    memPressure === "red" && "bg-rose-400/70",
                    memPressure === "unknown" && "bg-slate-600/40"
                  )}
                  style={{ width: memRatio ? `${Math.round(memRatio * 100)}%` : "10%" }}
                />
              </div>

              <div className="flex items-center justify-between">
                <span>Swap</span>
                <span className={swapUsed ? "text-rose-200" : "text-emerald-200"}>
                  {swapUsed == null ? "—" : `${formatBytes(swapUsed)}${swapUsed ? " in use" : " ok"}`}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center justify-between">
                  <span>CPU</span>
                  <span className="text-slate-100">
                    {cpuPercent != null
                      ? `${cpuPercent.toFixed(1)}%`
                      : "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>GPU</span>
                  <span className="text-slate-100">
                    {gpuUtil != null
                      ? `${gpuUtil.toFixed(0)}%`
                      : "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Metal</span>
                  <span
                    className={cx(
                      "text-slate-100",
                      metalSupported === false && "text-rose-200",
                      metalSupported === true && "text-emerald-200"
                    )}
                  >
                    {metalSupported == null
                      ? "Unknown"
                      : metalSupported
                        ? "Active"
                        : "Off"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>Context</span>
                  <span className="text-slate-100">
                    {contextChars != null ? `${contextChars}` : "—"}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="flex items-center justify-between">
                  <span>TPS</span>
                  <span className="text-slate-100">
                    {tps != null ? tps : "—"}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span>TTFT</span>
                  <span className="text-slate-100">
                    {ttft != null ? `${ttft} ms` : "—"}
                  </span>
                </div>
              </div>

              <div className="pt-1 text-[10px] text-slate-400">
                Model {modelName ?? "—"} · Quant {modelQuant ?? "—"} · {modelBackend ?? "—"}
              </div>
            </div>
          </div>
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
          <div className="space-y-3">
            <AnimatePresence>
              {messages.map((m) => {
                const formatted = m.role === "assistant" ? formatAssistantContent(m) : m.content;
                return (
                  <motion.div
                    key={m.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: 10 }}
                    className={cx(
                      "rounded-3xl overflow-hidden message-shell",
                      m.role === "user"
                        ? "ml-auto message-user"
                        : "mr-auto message-ai"
                    )}
                  >
                    {m.role === "assistant" ? (
                      <div className="p-4 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <div className="font-semibold">Jarvis</div>
                          {m.mode && (
                            <span className="text-xs text-slate-400">
                              {m.mode}
                            </span>
                          )}
                        </div>
                      </div>
                    ) : null}

                    <div className={cx("px-5 pb-5", m.role === "user" && "pt-4 text-left")}>
                      {m.role === "assistant" ? (
                        m.isLoading ? (
                          <div className="flex items-center gap-3 text-sm text-slate-300">
                            <div className="loading-dots" aria-hidden="true">
                              <span className="loading-dot" />
                              <span className="loading-dot" />
                              <span className="loading-dot" />
                            </div>
                            <span className="text-[11px] uppercase tracking-[0.3em] text-emerald-200/70">
                              Loading
                            </span>
                          </div>
                        ) : (
                          <div className="max-w-none text-[15px] leading-relaxed text-slate-100/95">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                table({ children, ...props }) {
                                  return (
                                    <div className="my-4 w-full overflow-x-auto rounded-xl border border-white/10">
                                      <table className="w-full border-collapse text-sm" {...props}>
                                        {children}
                                      </table>
                                    </div>
                                  );
                                },
                                thead({ children, ...props }) {
                                  return (
                                    <thead className="bg-white/5" {...props}>
                                      {children}
                                    </thead>
                                  );
                                },
                                th({ children, ...props }) {
                                  return (
                                    <th className="px-3 py-2 text-left font-semibold border-b border-white/10" {...props}>
                                      {children}
                                    </th>
                                  );
                                },
                                td({ children, ...props }) {
                                  return (
                                    <td className="px-3 py-2 align-top border-b border-white/5" {...props}>
                                      {children}
                                    </td>
                                  );
                                },
                                code({ className, children, ...props }) {
                                  const isBlock = typeof className === "string" && className.includes("language-");
                                  if (!isBlock) {
                                    return (
                                      <code className="rounded-md bg-white/5 px-1.5 py-0.5 text-[0.95em] font-mono" {...props}>
                                        {children}
                                      </code>
                                    );
                                  }
                                  return (
                                    <pre className="my-4 overflow-x-auto rounded-xl bg-black/40 p-4 text-sm border border-white/10 font-mono">
                                      <code className={className} {...props}>
                                        {children}
                                      </code>
                                    </pre>
                                  );
                                },
                              }}
                            >
                              {formatted}
                            </ReactMarkdown>

                          </div>
                        )
                      ) : (
                        <div className="whitespace-pre-wrap text-sm text-slate-100 leading-relaxed">
                          {m.content}
                        </div>
                      )}

                      {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                        <div className="mt-5">
                          <details className="group">
                            <summary className="cursor-pointer list-none flex items-center justify-between rounded-2xl border border-slate-800/60 bg-black/50 px-4 py-3 hover:bg-black/70 transition">
                              <span className="text-sm font-semibold">
                                Sources ({m.sources.length})
                              </span>
                              <span className="text-xs text-zinc-400 group-open:rotate-180 transition-transform">
                                ▾
                              </span>
                            </summary>

                            <div className="mt-3 grid gap-2">
                              {m.sources.map((s) => {
                                const isWeb = !!s.url;

                                return (
                                  <div key={s.label} className="rounded-xl border border-slate-700/50 p-3 glass-panel glass-panel--thin">
                                    <div className="text-xs font-semibold text-slate-300/70">
                                      {s.label} {isWeb ? "· Web" : "· Local"}
                                    </div>

                                    {isWeb ? (
                                      <div className="mt-1">
                                        <a
                                          href={s.url}
                                          target="_blank"
                                          rel="noreferrer"
                                          className="text-sm font-medium text-sky-300 hover:underline"
                                        >
                                          {s.title || s.url}
                                        </a>
                                        {s.snippet ? (
                                          <div className="mt-1 text-sm text-slate-300/80">
                                            {s.snippet}
                                          </div>
                                        ) : null}
                                      </div>
                                    ) : (
                                      <div className="mt-1 text-sm text-slate-300/80">
                                        <div className="truncate">{s.doc_path}</div>
                                        <div className="text-xs text-slate-400/80">
                                          chunk {s.chunk_index} {typeof s.score === "number" ? `· score ${s.score.toFixed(3)}` : ""}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                );
                              })}

                            </div>
                          </details>
                        </div>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        </div>

        {/* input at bottom */}
        <div className="rounded-3xl h-29 mt-2 mb-8 mx-auto w-full max-w-450 glass-panel glass-panel--input neon-edge">
          <div className="p-4">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder='Ask Jarvis something, or click the microphone to speak...'
              className="w-full min-h-5 resize-none bg-transparent text-sm leading-relaxed text-slate-100 outline-none placeholder:text-slate-500 font-mono"
            />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <div className="relative" ref={modeMenuRef}>
                  <button
                    type="button"
                    onClick={() => setShowModeMenu((v) => !v)}
                    ref={modeButtonRef}
                    className={cx(
                      "mt-1.5 h-8 w-8 rounded-2xl text-slate-200 transition hover:border-emerald-300/60 hover:bg-emerald-500/10 active:scale-[0.97] active:translate-y-px",
                      showModeMenu ? "border-emerald-300/70 bg-emerald-500/10" : "border-slate-800/70"
                    )}
                    title="Choose mode"
                  >
                    <span className="sr-only">Choose mode</span>
                    <svg
                      className="mx-auto h-4.5 w-4.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M12 5v14" />
                      <path d="M5 12h14" />
                    </svg>
                  </button>

                </div>
                {showModeMenu && modeMenuPos
                  ? createPortal(
                    <div
                      className="fixed z-9999 w-60 rounded-2xl glass-panel glass-panel--menu"
                      style={{ top: modeMenuPos.top, left: modeMenuPos.left, backgroundColor: "rgba(5,8,12,0.82)", backdropFilter: "blur(16px)" }}
                      ref={modeMenuPortalRef}
                    >
                      <div className="px-4 py-3 text-[11px] uppercase tracking-[0.35em] text-slate-400">
                        Modes
                      </div>
                      {(["local", "general", "search"] as Mode[]).map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => {
                            setMode(opt);
                            setShowModeMenu(false);
                          }}
                          className={cx(
                            "flex w-full items-center justify-between px-4 py-3 text-sm font-semibold transition",
                            opt === mode
                              ? "text-emerald-200 bg-emerald-500/10"
                              : "text-slate-200 hover:bg-slate-800/60"
                          )}
                        >
                          <span className="flex items-center gap-3">
                            <span className="h-2.5 w-2.5 rounded-full bg-linear-to-r from-emerald-300 to-teal-300" />
                            {modeLabels[opt]}
                          </span>
                          {opt === mode ? (
                            <svg
                              className="h-4 w-4 text-emerald-200"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="1.8"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              aria-hidden="true"
                            >
                              <path d="M5 12l4 4L19 7" />
                            </svg>
                          ) : null}
                        </button>
                      ))}
                    </div>,
                    document.body
                  )
                  : null}

                <div
                  className={cx(
                    "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold",
                    mode === "general" && "border-emerald-500/40 bg-emerald-500/15 text-emerald-100",
                    mode === "local" && "border-teal-500/40 bg-teal-500/15 text-teal-100",
                    mode === "search" && "border-cyan-400/40 bg-cyan-500/15 text-cyan-100"
                  )}
                >
                  {mode === "general" && (
                    <svg
                      className="h-4 w-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <circle cx="12" cy="12" r="9" />
                      <path d="M3 12h18" />
                      <path d="M12 3a15 15 0 0 1 0 18" />
                      <path d="M12 3a15 15 0 0 0 0 18" />
                    </svg>
                  )}
                  {mode === "local" && (
                    <svg
                      className="h-4 w-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M4 7a2 2 0 0 1 2-2h5l2 2h5a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z" />
                    </svg>
                  )}
                  {mode === "search" && (
                    <svg
                      className="h-4 w-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <circle cx="11" cy="11" r="7" />
                      <path d="M20 20l-3.5-3.5" />
                    </svg>
                  )}
                  <span className="text-[13px]">{modeLabels[mode]}</span>
                </div>

              </div>

              <div className="flex items-center gap-2">
                {/* Mic */}
                <button
                  onClick={toggleRecording}
                  disabled={loading}
                  className={cx(
                    "h-9 w-10 rounded-2xl border text-lg font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-px active:scale-[0.98]",
                    isRecording
                      ? "border-rose-400/70 bg-rose-500/15 text-rose-100 hover:border-rose-300/70 hover:bg-rose-500/20"
                      : "border-slate-800/70 bg-black/40 text-slate-200 hover:border-emerald-300/60 hover:bg-emerald-500/10"
                  )}
                  title={isRecording ? "Stop recording" : "Start recording"}
                >
                  {isRecording ? (
                    <span className="inline-flex h-3.5 w-3.5 rounded-sm bg-rose-200" />
                  ) : (
                    <svg
                      className="mx-auto h-4.5 w-4.5"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M12 3a3 3 0 0 0-3 3v5a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z" />
                      <path d="M19 11a7 7 0 0 1-14 0" />
                      <path d="M12 18v3" />
                      <path d="M8 21h8" />
                    </svg>
                  )}
                </button>

                {/* Mute */}
                <button
                  onClick={() => {
                    setTtsMuted((m) => {
                      const next = !m;
                      if (typeof window !== "undefined" && "speechSynthesis" in window) {
                        window.speechSynthesis.cancel();
                      }
                      spokenRef.current = "";
                      return next;
                    });
                  }}
                  className={cx(
                    "h-9 w-9 rounded-2xl border text-lg font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-px active:scale-[0.98]",
                    ttsMuted
                      ? "border-slate-800/70 bg-black/40 text-slate-200 hover:border-emerald-300/60 hover:bg-emerald-500/10"
                      : "border-emerald-300/60 bg-emerald-500/10 text-emerald-100"
                  )}
                  title={ttsMuted ? "Unmute voice" : "Mute voice"}
                >
                  {ttsMuted ? (
                    <svg
                      className="mx-auto h-4 w-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M11 5L6 9H3v6h3l5 4z" />
                      <path d="M19 9l-4 4 4 4" />
                    </svg>
                  ) : (
                    <svg
                      className="mx-auto h-4 w-4"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <path d="M11 5L6 9H3v6h3l5 4z" />
                      <path d="M15 9a3 3 0 0 1 0 6" />
                    </svg>
                  )}
                </button>

                {/* Clear */}
                <button
                  onClick={clearChat}
                  className="h-9 w-9 rounded-2xl border border-slate-800/70 bg-black/40 text-slate-200 transition duration-200 hover:border-rose-400/60 hover:bg-rose-500/10 active:translate-y-px active:scale-[0.98]"
                  title="Clear chat"
                >
                  <svg
                    className="mx-auto h-4 w-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M3 6h18" />
                    <path d="M8 6V4h8v2" />
                    <path d="M6 6l1 14h10l1-14" />
                  </svg>
                </button>

                <button
                  onClick={ask}
                  disabled={!canAsk}
                  className={cx(
                    "w-20 mb-1 rounded-2xl px-3 py-2 text-sm font-semibold transition relative overflow-hidden active:translate-y-px active:scale-[0.99]",
                    canAsk ? "text-white" : "bg-slate-800 text-slate-500 cursor-not-allowed"
                  )}
                >
                  {canAsk && (
                    <span className="absolute inset-0 bg-linear-to-r from-emerald-600 via-teal-500 to-cyan-400 opacity-90" />
                  )}
                  <span className="relative flex items-center justify-center gap-2">
                    {loading ? (
                      <span className="inline-flex items-center gap-2">
                        <span className="h-4 w-4 animate-spin rounded-full border border-white/40 border-t-white/90" />
                      </span>
                    ) : (
                      <>
                        <svg
                          className="h-4 w-4"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.6"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden="true"
                        >
                          <path d="M21 12L3 4l4 8-4 8 18-8z" />
                          <path d="M7 12h8" />
                        </svg>
                        <span>Ask</span>
                      </>
                    )}
                  </span>
                </button>
              </div>
            </div>

            {/* mic error + option */}
            <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
              <div className="min-h-4">
                {micError ? <span className="text-rose-300">{micError}</span> : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
