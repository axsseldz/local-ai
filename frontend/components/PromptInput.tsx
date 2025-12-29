"use client";

import { JSX, useEffect, useMemo, useRef, useState, ChangeEvent, KeyboardEvent, RefObject } from "react";
import { createPortal } from "react-dom";
import { cx } from "@/lib/cx";
import { Mode } from "@/types/chat";

type Props = {
  question: string;
  mode: Mode;
  modeLabels: Record<Mode, string>;
  modeIcons: Record<Mode, () => JSX.Element>;
  renderPromptPreview: (text: string) => JSX.Element | JSX.Element[];
  onQuestionChange: (e: ChangeEvent<HTMLTextAreaElement>) => void;
  onKeyDown: (e: KeyboardEvent<HTMLTextAreaElement>) => void;
  toggleRecording: () => void;
  isRecording: boolean;
  ttsMuted: boolean;
  onToggleMute: () => void;
  loading: boolean;
  canAsk: boolean;
  ask: () => void;
  clearChat: () => void;
  micError: string;
  setMode: (mode: Mode) => void;
  questionInputRef: RefObject<HTMLTextAreaElement>;
};

export function PromptInput({
  question,
  mode,
  modeLabels,
  modeIcons,
  renderPromptPreview,
  onQuestionChange,
  onKeyDown,
  toggleRecording,
  isRecording,
  ttsMuted,
  onToggleMute,
  loading,
  canAsk,
  ask,
  clearChat,
  micError,
  setMode,
  questionInputRef,
}: Props) {
  const [showModeMenu, setShowModeMenu] = useState(false);
  const [modeMenuPos, setModeMenuPos] = useState<{ top: number; left: number } | null>(null);
  const modeMenuRef = useRef<HTMLDivElement | null>(null);
  const modeButtonRef = useRef<HTMLButtonElement | null>(null);
  const modeMenuPortalRef = useRef<HTMLDivElement | null>(null);
  const promptOverlayRef = useRef<HTMLDivElement | null>(null);

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
    if (!showModeMenu) return;
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node;
      const hitMode = modeMenuRef.current?.contains(target) || modeMenuPortalRef.current?.contains(target);
      if (!hitMode) setShowModeMenu(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [showModeMenu]);

  const menuContent = useMemo(
    () => (
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
                      ? "text-cyan-100 bg-cyan-500/15"
                      : "text-slate-200 hover:bg-slate-800/60"
                  )}
                >
                  <span className="flex items-center gap-3">
                    {modeIcons[opt]()}
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
            "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold border-cyan-400/40 bg-cyan-500/15 text-cyan-100"
          )}
        >
          {modeIcons[mode]()}
          <span className="text-[13px]">{modeLabels[mode]}</span>
        </div>

      </div>
    ),
    [mode, modeIcons, modeLabels, showModeMenu, modeMenuPos]
  );

  return (
    <div className="rounded-3xl h-29 mt-2 mb-8 mx-auto w-full max-w-450 glass-panel glass-panel--input neon-edge">
      <div className="p-4">
        <div className="relative">
          <div
            ref={promptOverlayRef}
            aria-hidden
            className="pointer-events-none absolute inset-0 box-border overflow-hidden whitespace-pre-wrap wrap-break-word bg-transparent text-sm leading-relaxed text-slate-100 font-mono p-0"
          >
            {question ? (
              renderPromptPreview(question)
            ) : (
              <span className="text-slate-500">
                Ask Jarvis something, or click the microphone to speak...
              </span>
            )}
          </div>
          <textarea
            ref={questionInputRef}
            value={question}
            onChange={onQuestionChange}
            onKeyDown={onKeyDown}
            onScroll={(e) => {
              if (promptOverlayRef.current) {
                promptOverlayRef.current.scrollTop = e.currentTarget.scrollTop;
                promptOverlayRef.current.scrollLeft = e.currentTarget.scrollLeft;
              }
            }}
            placeholder="Ask Jarvis something, or click the microphone to speak..."
            className="w-full min-h-5 resize-none bg-transparent text-sm leading-relaxed text-transparent caret-slate-100 outline-none placeholder:text-transparent font-mono box-border p-0"
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          {menuContent}

          <div className="flex items-center gap-2">
            {/* Mic */}
            <button
              onClick={toggleRecording}
              disabled={loading}
              className={cx(
                "h-9 w-10 rounded-2xl text-lg font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-px active:scale-[0.98]",
                isRecording
                  ? "bg-cyan-500/20 text-cyan-100 hover:bg-cyan-500/30"
                  : "bg-cyan-500/10 text-cyan-100 hover:bg-cyan-500/20"
              )}
              title={isRecording ? "Stop recording" : "Start recording"}
            >
              {isRecording ? (
                <span className="inline-flex h-3.5 w-3.5 rounded-sm bg-cyan-200" />
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
              onClick={onToggleMute}
              className={cx(
                "h-9 w-10 rounded-2xl text-lg font-semibold transition duration-200 focus:outline-none focus-visible:outline-none focus:ring-0 focus:ring-offset-0 active:translate-y-px active:scale-[0.98]",
                ttsMuted
                  ? "bg-cyan-500/10 text-cyan-200/80 hover:bg-cyan-500/30"
                  : "bg-cyan-500/10 text-cyan-100 hover:bg-cyan-500/20"
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
              className="h-9 w-10 rounded-2xl bg-cyan-500/10 text-cyan-100 transition duration-200 hover:bg-cyan-500/20 active:translate-y-px active:scale-[0.98]"
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

        <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
          <div className="min-h-4">
            {micError ? <span className="text-rose-300">{micError}</span> : null}
          </div>
        </div>
      </div>
    </div>
  );
}
