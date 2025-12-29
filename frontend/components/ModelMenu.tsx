"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cx } from "@/lib/cx";

type ModelMenuProps = {
  model: string;
  modelOptions: string[];
  onChange: (model: string) => void;
};

export function ModelMenu({ model, modelOptions, onChange }: ModelMenuProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) {
      setPos(null);
      return;
    }

    const updatePos = () => {
      const btn = buttonRef.current;
      if (!btn || typeof window === "undefined") return;
      const rect = btn.getBoundingClientRect();
      const menuWidth = 240;
      const left = Math.max(8, Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - 8));
      const top = rect.bottom + 8;
      setPos({ top, left });
    };

    updatePos();
    window.addEventListener("resize", updatePos);
    window.addEventListener("scroll", updatePos, true);
    return () => {
      window.removeEventListener("resize", updatePos);
      window.removeEventListener("scroll", updatePos, true);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      const target = e.target as Node;
      const hitButton = buttonRef.current?.contains(target);
      const hitMenu = menuRef.current?.contains(target);
      if (!hitButton && !hitMenu) setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  return (
    <div className="relative" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        ref={buttonRef}
        className={cx(
          "inline-flex h-11 items-center gap-2 rounded-2xl border border-slate-800/70 bg-transparent px-4 text-[12px] font-semibold text-slate-200 transition hover:border-cyan-300/60 hover:text-cyan-100",
          open && "border-cyan-300/70 bg-cyan-500/15 text-cyan-100"
        )}
        title="Choose model"
      >
        <span className="max-w-44 truncate">Model: {model}</span>
        <svg
          className={cx("h-3.5 w-3.5 text-cyan-200 transition-transform", open && "rotate-180")}
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

      {open && pos && createPortal(
        <div
          className="fixed z-9999 w-60 overflow-hidden rounded-xl glass-panel glass-panel--menu"
          style={{ top: pos.top, left: pos.left }}
        >
          <div className="px-3 py-2 text-[11px] uppercase tracking-[0.35em] text-slate-400">
            Models
          </div>
          <div className="py-1">
            {modelOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                onClick={() => {
                  onChange(opt);
                  setOpen(false);
                }}
                className={cx(
                  "flex w-full items-center justify-between px-3 py-2 text-xs font-semibold transition",
                  opt === model
                    ? "text-cyan-100 bg-cyan-500/15"
                    : "text-slate-200 hover:bg-slate-800/60"
                )}
              >
                <span className="truncate">{opt}</span>
                {opt === model ? (
                  <svg
                    className="h-3.5 w-3.5 text-cyan-200"
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
          </div>
        </div>,
        document.body
      )}
    </div>
  );
}
