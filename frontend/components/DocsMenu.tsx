"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cx } from "@/lib/cx";
import { DocItem } from "@/types/chat";

type DocsMenuProps = {
  docs: DocItem[];
  label?: string;
};

export function DocsMenu({ docs, label = "Documents" }: DocsMenuProps) {
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
      const menuWidth = 224;
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
    <div className="relative z-50" ref={menuRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        ref={buttonRef}
        className={cx(
          "inline-flex items-center gap-2 rounded-lg border bg-transparent px-2 py-1 text-[11px] font-semibold text-slate-200 transition",
          open
            ? "border-emerald-400/60 bg-emerald-500/15 text-emerald-100"
            : "border-slate-800/70 hover:border-emerald-300/60 hover:text-emerald-100"
        )}
        title="Documents in data/documents"
      >
        <span className="max-w-30 truncate">{label}</span>
        <svg
          className={cx("h-3 w-3 text-emerald-200 transition-transform", open && "rotate-180")}
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
          className="fixed z-9999 w-56 overflow-hidden rounded-xl glass-panel glass-panel--menu"
          style={{ top: pos.top, left: pos.left }}
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
      )}
    </div>
  );
}
