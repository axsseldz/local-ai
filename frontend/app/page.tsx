"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnimatePresence, motion } from "framer-motion";

type Mode = "auto" | "local" | "general";

type Source = {
  label: string;
  doc_path: string;
  chunk_index: number;
  score?: number | null;
};

type AskResponse = {
  answer: string;
  mode: string;
  task?: string;
  sources?: Source[];
  used_tools?: string[];
  model?: string;
  routing_reason?: string;
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
  routing_reason?: string;
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

// downsample to 16kHz mono (simple + efectivo para MVP)
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
  const [mode, setMode] = useState<Mode>("auto");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexStarting, setIndexStarting] = useState(false);
  const [docs, setDocs] = useState<DocItem[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string>(""); // filename
  const [uploading, setUploading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [micError, setMicError] = useState("");
  const [ttsEnabled, setTtsEnabled] = useState(true);

  const spokenRef = useRef<string>(""); // avoid double-speaking
  const finalTranscriptRef = useRef<string>("");
  const chatRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  const [liveTranscript, setLiveTranscript] = useState(""); // texto parcial live


  useEffect(() => {
    const id = requestAnimationFrame(() => {
      chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(id);
  }, [messages]);

  useEffect(() => {
    fetchIndexStatus();

    const t = setInterval(() => {
      // poll faster while running
      const running = indexStatus?.is_indexing;
      if (running) fetchIndexStatus();
    }, 1200);

    const slow = setInterval(() => {
      // poll slow when idle
      const running = indexStatus?.is_indexing;
      if (!running) fetchIndexStatus();
    }, 8000);

    return () => {
      clearInterval(t);
      clearInterval(slow);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexStatus?.is_indexing]);


  const canAsk = useMemo(() => question.trim().length > 0 && !loading, [question, loading]);


  async function ask() {
    await askWithText(question);
  }

  function speak(text: string) {
    if (!ttsEnabled) return;
    if (!("speechSynthesis" in window)) return;

    // Prevent re-speaking same message
    if (spokenRef.current === text) return;
    spokenRef.current = text;

    window.speechSynthesis.cancel(); // stop previous speech

    const utterance = new SpeechSynthesisUtterance(text);

    // Optional tuning (Mac voices sound great)
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Prefer a natural English voice
    const voices = window.speechSynthesis.getVoices();
    const preferred =
      voices.find(v => v.name.includes("Samantha")) ||
      voices.find(v => v.lang.startsWith("en")) ||
      voices[0];

    if (preferred) utterance.voice = preferred;

    window.speechSynthesis.speak(utterance);
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

      // refresh docs + index status soon
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
      // ignore (backend may be down)
    }
  }

  async function startIndexNow() {
    try {
      setIndexStarting(true);
      await fetch("http://localhost:8000/index/run", { method: "POST" });
      // refresh quickly after starting
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

      // 1) WS connect
      const ws = new WebSocket("ws://localhost:8000/voice/stt/ws");
      wsRef.current = ws;

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "partial") {
            setLiveTranscript(msg.text || "");
            // “preview” en textarea mientras hablas:
            setQuestion(msg.text || "");
          } else if (msg.type === "final") {
            const finalText = (msg.text || "").trim();
            if (finalText) {
              finalTranscriptRef.current = finalText; // ✅ store final transcript
              setQuestion(finalText);                 // ✅ show it in textarea
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

      // 2) Audio graph -> ScriptProcessor -> PCM16 -> WS
      const AudioCtx = (window.AudioContext || (window as any).webkitAudioContext);
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      sourceRef.current = source;

      // bufferSize 4096 es ok para MVP
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const wsNow = wsRef.current;
        if (!wsNow || wsNow.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0); // Float32
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
  const textToSend = (finalTranscriptRef.current || question).trim();
  finalTranscriptRef.current = ""; // reset for next recording

  function stopRecording() {
    if (!isRecording) return;
    setIsRecording(false);

    // stop audio graph
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();

    processorRef.current = null;
    sourceRef.current = null;

    // close audio context
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => { });
      audioCtxRef.current = null;
    }

    // close websocket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
    }
    wsRef.current = null;

    // clear partial
    setLiveTranscript("");

    if (textToSend) {
      setTimeout(() => askWithText(textToSend), 50);
    }
  }


  function toggleRecording() {
    if (isRecording) stopRecording();
    else startRecording();
  }

  // Small helper so voice can send without relying on state timing
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
      content: "Thinking…",
      createdAt: Date.now(),
    };

    spokenRef.current = ""; // allow speaking for this new answer
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

      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: "" } : m)));

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let assistantText = ""; // ✅ accumulate streamed answer for TTS

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
            const finalText = assistantText.trim();
            if (finalText) {
              speak(finalText); // 🔊 THIS WILL NOW WORK
            }
            continue;
          };

          const chunk = data;
          if (!metaApplied) metaApplied = true;

          // accumulate locally for TTS
          assistantText += chunk;

          // update UI
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: (m.content || "") + chunk }
                : m
            )
          );
        }
      }
    } catch {
      const msg = "Could not reach backend. Is FastAPI running on http://localhost:8000 ?";
      setError(msg);
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: msg, error: true } : m)));
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter newline
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

  // Refresh docs list when indexing finishes successfully
  useEffect(() => {
    if (indexStatus?.state === "ok" && !indexStatus?.is_indexing) {
      fetchDocs();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [indexStatus?.last_finished_at]);

  const controlButtonBase =
    "relative overflow-hidden rounded-xl border px-3 py-1.5 text-xs font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-[1px]";
  const controlButtonActive =
    "border-slate-800/70 bg-slate-950/40 text-slate-200 hover:border-cyan-400/60 hover:bg-cyan-500/10 hover:shadow-[0_12px_45px_rgba(34,211,238,0.18)] active:scale-[0.98] active:border-cyan-300/70 active:bg-cyan-500/15 cursor-pointer";
  const controlButtonDisabled = "border-slate-800/70 bg-slate-950/30 text-slate-500 cursor-not-allowed shadow-none";

  return (
    <div
      className="app-shell relative min-h-screen bg-[#0d1218] text-slate-100 overflow-y-scroll flex flex-col"
      style={{ scrollbarGutter: "stable" }}
    >
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-32 -left-32 w-130 rounded-full blur-3xl opacity-22 bg-linear-to-r from-[#113042] via-[#0f2330] to-[#0b141e]" />
        <div className="absolute -bottom-40 -right-40 h-155 w-155 rounded-full blur-3xl opacity-16 bg-linear-to-r from-[#0c1a26] via-[#0a131c] to-[#05090f]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.015),transparent_60%)]" />
        <div className="absolute inset-0 opacity-[0.06] bg-[radial-gradient(circle_at_20%_20%,rgba(45,212,191,0.1),transparent_25%),radial-gradient(circle_at_80%_12%,rgba(125,211,252,0.1),transparent_22%),radial-gradient(circle_at_40%_78%,rgba(94,234,212,0.1),transparent_28%)]" />
        <div className="absolute inset-0 opacity-[0.03] bg-[linear-gradient(90deg,rgba(255,255,255,0.08)_1px,transparent_1px),linear-gradient(0deg,rgba(255,255,255,0.08)_1px,transparent_1px)] bg-size-[120px_120px]" />
      </div>

      <div className="absolute top-4 left-4 z-20 flex items-center gap-3">
        <div className="orb-container static-orb w-8 h-8">
          <div className="orb inner-anim" />
        </div>
        <div className="flex flex-col">
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-2xl font-semibold jarvis-anim leading-tight"
          >
            Jarvis
          </motion.div>
          <p className="text-xs text-slate-400">Local + general AI</p>
        </div>
      </div>

      <div className="relative mx-auto max-w-5xl px-4 pt-2 pb-4 flex-1 flex flex-col gap-3 min-h-0">
        {/* error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="rounded-2xl border border-red-900/50 bg-red-950/30 p-4 text-sm text-red-300 backdrop-blur"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* chat history above prompt */}
        <div
          className="mx-auto w-full max-w-5xl rounded-3xl border border-slate-800/60 bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a] p-4 mt-4 flex-1 min-h-80 min-w-0 overflow-y-auto scroll-smooth shadow-[0_18px_80px_rgba(0,0,0,0.35)] backdrop-blur-xl"
          ref={chatRef}
          style={{ scrollbarGutter: "stable" }}
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
          title="Drag & drop a file here to upload + index"
        >
          <div className="space-y-3">
            <AnimatePresence>
              {messages.map((m) => (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  className={cx(
                    "rounded-3xl border border-slate-800/60 backdrop-blur-xl shadow-[0_18px_80px_rgba(0,0,0,0.35)] overflow-hidden",
                    m.role === "user"
                      ? "ml-auto bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a]"
                      : "mr-auto bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a]"
                  )}
                >
                  <div className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className={cx(
                          "h-2.5 w-2.5 rounded-full shadow-[0_0_18px_rgba(56,189,248,0.35)]",
                          m.role === "user"
                            ? "bg-linear-to-r from-emerald-300 to-cyan-400"
                            : "bg-linear-to-r from-slate-300 to-slate-500"
                        )}
                      />
                      <div className="font-semibold">
                        {m.role === "user" ? "You" : "Jarvis"}
                      </div>
                      {m.role === "assistant" && (m.mode || m.task) && (
                        <span className="text-xs text-slate-400">
                          {m.mode ? m.mode : ""}
                          {m.mode && m.task ? " • " : ""}
                          {m.task ? m.task : ""}
                        </span>
                      )}
                    </div>

                    {m.role === "assistant" && !m.error && m.content && m.content !== "Thinking…" && (
                      <button
                        className="rounded-xl border border-slate-800/70 bg-slate-950/40 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-950/70 transition"
                        onClick={() => navigator.clipboard.writeText(m.content)}
                        title="Copy answer"
                      >
                        Copy
                      </button>
                    )}
                  </div>

                  <div className="px-5 pb-5">
                    {m.role === "assistant" ? (
                      <div className="prose prose-invert max-w-none leading-relaxed prose-p:my-3 prose-li:my-1">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap text-sm text-slate-100 leading-relaxed">
                        {m.content}
                      </div>
                    )}

                    {m.role === "assistant" && m.sources && m.sources.length > 0 && (
                      <div className="mt-5">
                        <details className="group">
                          <summary className="cursor-pointer list-none flex items-center justify-between rounded-2xl border border-slate-800/60 bg-slate-950/50 px-4 py-3 hover:bg-slate-950/70 transition">
                            <span className="text-sm font-semibold">
                              Sources ({m.sources.length})
                            </span>
                            <span className="text-xs text-zinc-400 group-open:rotate-180 transition-transform">
                              ▾
                            </span>
                          </summary>

                          <div className="mt-3 grid gap-2">
                            {m.sources.map((s) => (
                              <div
                                key={`${m.id}-${s.doc_path}-${s.chunk_index}-${s.label}`}
                                className="rounded-2xl border border-slate-800/60 bg-slate-950/50 px-4 py-3"
                              >
                                <div className="flex items-center justify-between">
                                  <div className="text-sm font-semibold">{s.label}</div>
                                  {typeof s.score === "number" && (
                                    <div className="text-xs text-zinc-400">
                                      score {s.score.toFixed(3)}
                                    </div>
                                  )}
                                </div>
                                <div className="mt-1 text-xs text-zinc-300 break-all">
                                  {s.doc_path}
                                </div>
                                <div className="mt-1 text-xs text-zinc-400">
                                  chunk {s.chunk_index}
                                </div>
                              </div>
                            ))}
                          </div>
                        </details>
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* input at bottom */}
        <div className="rounded-3xl h-35 border border-slate-800/60 bg-linear-to-r from-[#0f1823] via-[#0f1f2b] to-[#0b111a] backdrop-blur-xl shadow-[0_18px_80px_rgba(0,0,0,0.45)] mt-2 mb-1 mx-auto w-full max-w-5xl">
          <div className="p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="relative">
                <select
                  className="select-modern appearance-none rounded-xl border border-slate-800/70 bg-linear-to-r from-[#0f1823] via-[#0f1c28] to-[#0b1420] px-4 py-2 pr-9 text-sm font-medium text-slate-100 shadow-sm focus:outline-none focus:ring-0 focus:border-cyan-500/40 hover:border-cyan-400/50 transition"
                  value={mode}
                  onChange={(e) => setMode(e.target.value as Mode)}
                >
                  <option className="bg-[#0f1620] text-slate-100" value="auto">
                    Auto
                  </option>
                  <option className="bg-[#0f1620] text-slate-100" value="local">
                    Local
                  </option>
                  <option className="bg-[#0f1620] text-slate-100" value="general">
                    General
                  </option>
                </select>
                <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-cyan-200">
                  ▾
                </span>
              </div>

              <div className="flex items-center gap-2">
                {/* Docs dropdown */}
                <div className="relative">
                  <select
                    className="select-modern appearance-none rounded-xl border border-slate-800/70 bg-linear-to-r from-[#0f1823] via-[#0f1c28] to-[#0b1420] px-3 py-1.5 pr-8 text-xs font-semibold text-slate-200 shadow-sm focus:outline-none focus:ring-0 focus:border-cyan-500/40 hover:border-cyan-400/50 transition"
                    value={selectedDoc}
                    onChange={(e) => setSelectedDoc(e.target.value)}
                    title="Documents in data/documents"
                  >
                    {docs.length === 0 ? (
                      <option className="bg-[#0f1620] text-slate-100" value="">
                        No docs
                      </option>
                    ) : (
                      docs.map((d) => (
                        <option key={d.name} className="bg-[#0f1620] text-slate-100" value={d.name}>
                          {d.name}
                        </option>
                      ))
                    )}
                  </select>
                  <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-cyan-200">
                    ▾
                  </span>
                </div>

                {/* Upload (file picker) */}
                <label
                  className={cx(
                    controlButtonBase,
                    uploading ? controlButtonDisabled : controlButtonActive
                  )}
                  title="Upload a document into data/documents and auto-index"
                >
                  {uploading ? "Uploading…" : "Upload"}
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

                {/* Index status pill */}
                <div
                  className={cx(
                    "rounded-xl border px-3 py-1.5 text-xs font-semibold",
                    indexStatus?.state === "running" && "border-cyan-500/30 text-cyan-200 bg-cyan-500/10",
                    indexStatus?.state === "ok" && "border-emerald-500/25 text-emerald-200 bg-emerald-500/10",
                    indexStatus?.state === "error" && "border-rose-500/25 text-rose-200 bg-rose-500/10",
                    (!indexStatus || indexStatus?.state === "idle") && "border-slate-800/70 text-slate-300 bg-slate-950/40"
                  )}
                  title={
                    indexStatus?.state === "error"
                      ? indexStatus?.last_error ?? "Index error"
                      : indexStatus?.stats
                        ? `files=${indexStatus.stats.files_on_disk}, indexed=${indexStatus.stats.indexed_files}, skipped=${indexStatus.stats.skipped_files}, deleted=${indexStatus.stats.deleted_files}`
                        : "Index status"
                  }
                >
                  {indexStatus?.is_indexing ? "Indexing…" : `Index: ${indexStatus?.state ?? "…"}`}
                </div>

                {/* Index now */}
                <button
                  onClick={startIndexNow}
                  disabled={!!indexStatus?.is_indexing || indexStarting}
                  className={cx(
                    controlButtonBase,
                    indexStatus?.is_indexing || indexStarting ? controlButtonDisabled : controlButtonActive
                  )}
                  title="Run indexing now"
                >
                  Index now
                </button>

                {/* Clear */}
                <button
                  onClick={clearChat}
                  className={cx(controlButtonBase, controlButtonActive)}
                  title="Clear chat"
                >
                  Clear
                </button>
              </div>

            </div>

            <div className="mt-2 flex items-end gap-3">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder='Try: "Summarize my resume in 3 bullets" or "How do computers work?"'
                className="flex-1 min-h-16 w-200 h-16 resize-y rounded-2xl border border-slate-800/70 bg-[#0d131c] px-4 py-3 text-sm leading-relaxed transition outline-none focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 focus:border-slate-800/70"
              />
              {/* Mic */}
              <button
                onClick={toggleRecording}
                disabled={loading}
                className={cx(
                  "h-14 w-14 mb-1 rounded-2xl border text-lg font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-px",
                  isRecording
                    ? "border-rose-400/70 bg-rose-500/15 text-rose-100 hover:border-rose-300/70 hover:bg-rose-500/20 hover:shadow-[0_0_30px_rgba(244,63,94,0.25)] active:scale-[0.98]"
                    : "border-slate-800/70 bg-slate-950/40 text-slate-200 hover:border-cyan-400/60 hover:bg-cyan-500/10 hover:shadow-[0_12px_45px_rgba(34,211,238,0.18)] active:scale-[0.98] active:border-cyan-300/70 active:bg-cyan-500/15"
                )}
                title={isRecording ? "Stop recording" : "Start recording"}
              >
                {isRecording ? "■" : "🎤"}
              </button>

              <button
                onClick={() => {
                  setTtsEnabled(v => !v);
                  window.speechSynthesis.cancel(); // stop if muting mid-speech
                }}
                className="rounded-xl px-3 py-1.5 text-sm border border-zinc-300 dark:border-zinc-700
             bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition"
              >
                {ttsEnabled ? "🔊 Voice On" : "🔇 Muted"}
              </button>

              <button
                onClick={ask}
                disabled={!canAsk}
                className={cx(
                  "rounded-2xl px-4 py-2 h-14 mb-1 text-sm font-semibold transition relative overflow-hidden",
                  canAsk ? "text-white shadow-lg shadow-slate-900/40" : "bg-slate-800 text-slate-500 cursor-not-allowed"
                )}
              >
                {canAsk && <span className="absolute inset-0 bg-linear-to-r from-cyan-600 via-cyan-500 to-emerald-400 opacity-90" />}
                <span className={cx("relative", canAsk ? "drop-shadow" : "")}>{loading ? "Generating…" : "Ask"}</span>
              </button>
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
