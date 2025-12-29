"use client";

import { cx } from "@/lib/cx";
import { IndexStatus } from "@/types/chat";

type Props = {
  indexStatus: IndexStatus | null;
  indexProgress: number | null;
  indexStarting: boolean;
  onStartIndex: () => void;
};

export function IndexStatusCard({ indexStatus, indexProgress, indexStarting, onStartIndex }: Props) {
  const indexStats = indexStatus?.stats;
  return (
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

      <button
        onClick={onStartIndex}
        disabled={!!indexStatus?.is_indexing || indexStarting}
        className={cx(
          "mt-3 rounded-lg border px-2 py-1 text-[11px] font-semibold transition",
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
  );
}
